from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Sequence

try:
    from .env_file import load_local_env_files
except ImportError:
    from env_file import load_local_env_files

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


load_local_env_files()
apply_native_runtime_defaults()

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

try:
    from .ai_loop_engine import AILoopEngine, DocumentProcessingError
    from .document_config import MAX_DOCUMENT_BYTES
    from .thread_store import (
        DEFAULT_THREAD_TITLE,
        ThreadStore,
        safe_title,
    )
    from .loop_engine import DEFAULT_LOOP_RECIPE_ID
    from .web_contract import (
        APP_TITLE,
        TEXT_ENCODING_OPTIONS,
        empty_query_response_dict,
        env_flag,
        normalize_text_encoding,
        query_response_dict,
        runtime_status_dict,
        status_with_unexpected_upload_error,
        upload_status_message,
    )
except ImportError:
    from ai_loop_engine import AILoopEngine, DocumentProcessingError
    from document_config import MAX_DOCUMENT_BYTES
    from thread_store import (
        DEFAULT_THREAD_TITLE,
        ThreadStore,
        safe_title,
    )
    from loop_engine import DEFAULT_LOOP_RECIPE_ID
    from web_contract import (
        APP_TITLE,
        TEXT_ENCODING_OPTIONS,
        empty_query_response_dict,
        env_flag,
        normalize_text_encoding,
        query_response_dict,
        runtime_status_dict,
        status_with_unexpected_upload_error,
        upload_status_message,
    )


logging.basicConfig(
    level=logging.DEBUG if env_flag("APP_DEBUG", False) else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
MAX_UPLOAD_FILENAME_LENGTH = 180
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_SESSION_ID_LENGTH = 96
MAX_QUERY_HISTORY_MESSAGES = 12
MAX_SEMANTIC_MEMORY_MESSAGES = 4
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
THREAD_DB_PATH_ENV = "AI_LOOP_THREAD_DB_PATH"
qa_system: Optional[AILoopEngine] = None
thread_store_system: Optional[ThreadStore] = None


class QueryRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    recipe_id: Optional[str] = None
    context_provider: Optional[str] = None


class ClearChatRequest(BaseModel):
    session_id: Optional[str] = None


class ThreadCreateRequest(BaseModel):
    title: Optional[str] = None


class ThreadUpdateRequest(BaseModel):
    title: Optional[str] = None


class RecipeWriteRequest(BaseModel):
    recipe_id: Optional[str] = None
    name: str
    goal: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    success_criteria: Optional[list[str]] = None
    stop_condition: Optional[str] = None
    context_provider: Optional[str] = None
    model_profile: Optional[str] = None
    verifier: Optional[str] = None
    metadata: Optional[dict] = None


class RecipePatchRequest(BaseModel):
    name: Optional[str] = None
    goal: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    success_criteria: Optional[list[str]] = None
    stop_condition: Optional[str] = None
    context_provider: Optional[str] = None
    model_profile: Optional[str] = None
    verifier: Optional[str] = None
    metadata: Optional[dict] = None


def get_engine() -> AILoopEngine:
    global qa_system
    if qa_system is None:
        qa_system = AILoopEngine(fast_mode=env_flag("FAST_MODE", False))
    return qa_system


def default_thread_store_path() -> Path:
    configured_path = os.getenv(THREAD_DB_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".ai-loop-engine" / "threads.sqlite3"


def get_thread_store() -> ThreadStore:
    global thread_store_system
    if thread_store_system is None:
        thread_store_system = ThreadStore(default_thread_store_path())
    return thread_store_system


def safe_upload_name(filename: str | None) -> str:
    raw_name = filename or ""
    raw_basename = os.path.basename(raw_name)
    if any(ord(char) < 32 or ord(char) == 127 for char in raw_basename):
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    name = raw_basename.strip()
    if raw_name and not name:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    if name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    if len(name.encode("utf-8")) > MAX_UPLOAD_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    return name or "upload"


def safe_session_id(session_id: str | None) -> str:
    value = str(session_id or "default").strip()
    if not value:
        return "default"
    if len(value.encode("utf-8")) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid session id.")
    if not SESSION_ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid session id.")
    return value


def safe_path_id(value: str | None, *, label: str) -> str:
    record_id = str(value or "").strip()
    if (
        not record_id
        or len(record_id.encode("utf-8")) > MAX_SESSION_ID_LENGTH
        or not SESSION_ID_PATTERN.fullmatch(record_id)
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    return record_id


def title_from_message(message: str) -> str:
    return safe_title(message)


def thread_detail_response(thread) -> dict:
    return thread.detail_dict()


def thread_summary_response(thread) -> dict:
    return thread.summary_dict()


def loop_run_summary_response(run) -> dict:
    return run.summary_dict()


def loop_run_detail_response(run) -> dict:
    return run.detail_dict(public=True)


def recipe_summary_response(recipe) -> dict:
    return recipe.summary_dict()


def recipe_detail_response(recipe) -> dict:
    return recipe.to_dict()


def conversation_history_from_messages(messages) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ]


def semantic_memory_context_for_query(
    engine: AILoopEngine,
    store: ThreadStore,
    *,
    session_id: str,
    message: str,
    exclude_message_ids: Sequence[int],
) -> tuple[list[dict], str]:
    try:
        embedding_model = engine.memory_embedding_model_label()
        if not store.has_message_embeddings(session_id, embedding_model):
            return [], "not_requested"
        embedding_model, vectors = engine.embed_memory_texts([message])
        memories = store.semantic_memories(
            session_id,
            embedding_model=embedding_model,
            query_vector=vectors[0],
            limit=MAX_SEMANTIC_MEMORY_MESSAGES,
            exclude_message_ids=tuple(int(value) for value in exclude_message_ids),
        )
        return [memory.to_context_dict() for memory in memories], (
            "retrieved" if memories else "empty"
        )
    except Exception as exc:
        LOGGER.info(
            "Semantic thread memory unavailable for session %s: %s",
            session_id,
            exc.__class__.__name__,
        )
        return [], "unavailable"


def index_thread_memory(
    engine: AILoopEngine,
    store: ThreadStore,
    messages: Sequence[object],
) -> None:
    indexable_messages = [
        message
        for message in messages
        if getattr(message, "role", None) in {"user", "assistant"}
        and str(getattr(message, "content", "")).strip()
    ]
    if not indexable_messages:
        return
    try:
        embedding_model, vectors = engine.embed_memory_texts(
            [str(message.content) for message in indexable_messages]
        )
        for message, vector in zip(indexable_messages, vectors):
            store.upsert_message_embedding(
                message,
                embedding_model=embedding_model,
                vector=vector,
            )
    except Exception as exc:
        LOGGER.info(
            "Semantic thread memory indexing skipped: %s",
            exc.__class__.__name__,
        )


def clear_chat_history_for_session(engine: AILoopEngine, session_id: str) -> None:
    history = getattr(engine, "chat_history", None)
    if not isinstance(history, list):
        return

    def entry_session_id(entry: object) -> str:
        if isinstance(entry, dict):
            return str(entry.get("session_id") or "default").strip() or "default"
        return "default"

    history[:] = [
        entry for entry in history if entry_session_id(entry) != session_id
    ]


async def write_upload_file(upload: UploadFile, upload_path: Path) -> None:
    bytes_written = 0
    with upload_path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_DOCUMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded document exceeds the 25 MB limit.",
                )
            output.write(chunk)


def create_app(
    engine: Optional[AILoopEngine] = None,
    thread_store: Optional[ThreadStore] = None,
    engine_factory: Optional[Callable[[str], AILoopEngine]] = None,
) -> FastAPI:
    if engine is not None and engine_factory is not None:
        raise ValueError("Provide either engine or engine_factory, not both.")

    api = FastAPI(title=APP_TITLE)
    api.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    resolved_thread_store = (
        thread_store
        if thread_store is not None
        else (
            ThreadStore.in_memory()
            if engine is not None or engine_factory is not None
            else None
        )
    )
    scoped_runtimes: dict[str, AILoopEngine] = {}
    scoped_runtime_lock = threading.Lock()

    def build_runtime(session_id: str) -> AILoopEngine:
        if engine_factory is not None:
            return engine_factory(session_id)
        return AILoopEngine(fast_mode=env_flag("FAST_MODE", False))

    def runtime(session_id: str = "default") -> AILoopEngine:
        if engine is not None:
            return engine
        safe_id = str(session_id or "default").strip() or "default"
        with scoped_runtime_lock:
            current_runtime = scoped_runtimes.get(safe_id)
            if current_runtime is None:
                current_runtime = build_runtime(safe_id)
                scoped_runtimes[safe_id] = current_runtime
            return current_runtime

    def loaded_runtime(session_id: str = "default") -> Optional[AILoopEngine]:
        if engine is not None:
            return engine
        safe_id = str(session_id or "default").strip() or "default"
        with scoped_runtime_lock:
            return scoped_runtimes.get(safe_id)

    def drop_runtime(
        session_id: str,
        *,
        expected_runtime: Optional[AILoopEngine] = None,
    ) -> None:
        if engine is not None:
            return
        safe_id = str(session_id or "default").strip() or "default"
        with scoped_runtime_lock:
            current_runtime = scoped_runtimes.get(safe_id)
            if expected_runtime is None or current_runtime is expected_runtime:
                scoped_runtimes.pop(safe_id, None)

    def thread_instance_is_current(
        session_id: str,
        *,
        expected_instance_id: str,
    ) -> bool:
        thread = threads().get_thread(session_id)
        return (
            thread is not None
            and thread.instance_id == expected_instance_id
        )

    def stale_upload_response(
        session_id: str,
        *,
        upload_runtime: Optional[AILoopEngine] = None,
    ) -> JSONResponse:
        if upload_runtime is not None:
            drop_runtime(session_id, expected_runtime=upload_runtime)
        return JSONResponse(
            {
                "detail": (
                    "Thread changed before document upload completed. "
                    "Please upload the file again in the active thread."
                )
            },
            status_code=409,
        )

    def threads() -> ThreadStore:
        return resolved_thread_store if resolved_thread_store is not None else get_thread_store()

    @api.middleware("http")
    async def reject_oversized_upload_request(request: Request, call_next):
        if request.method.upper() == "POST" and request.url.path == "/api/documents":
            content_length = request.headers.get("content-length")
            if not content_length:
                return JSONResponse(
                    {"detail": "Content-Length is required for document uploads."},
                    status_code=411,
                )
            try:
                request_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length."},
                    status_code=400,
                )
            if request_bytes > MAX_DOCUMENT_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    {"detail": "Uploaded document exceeds the 25 MB limit."},
                    status_code=413,
                )
        response = await call_next(request)
        if request.method.upper() in {"GET", "HEAD"} and (
            request.url.path == "/" or request.url.path.startswith("/assets/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @api.get("/api/health")
    def health() -> dict:
        return {"ok": True, "app": APP_TITLE}

    @api.get("/api/config")
    def config() -> dict:
        return {
            "title": APP_TITLE,
            "default_recipe_id": DEFAULT_LOOP_RECIPE_ID,
            "text_encodings": [
                {"label": label, "value": value}
                for label, value in TEXT_ENCODING_OPTIONS.items()
            ],
        }

    @api.get("/api/status")
    def status(request: Request, session_id: Optional[str] = None) -> dict:
        requested_session_id = (
            session_id
            or request.headers.get("x-ai-loop-session-id")
            or request.headers.get("x-session-id")
        )
        safe_id = safe_session_id(requested_session_id)
        if requested_session_id and threads().get_thread(safe_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return runtime_status_dict(runtime(safe_id).status())

    @api.get("/api/threads")
    def list_threads() -> dict:
        store = threads()
        records = store.list_threads()
        if not records:
            records = [store.create_thread(title=DEFAULT_THREAD_TITLE)]
        return {"threads": [thread_summary_response(record) for record in records]}

    @api.post("/api/threads")
    def create_thread(request: Optional[ThreadCreateRequest] = Body(default=None)) -> dict:
        title = request.title if request else None
        return thread_detail_response(threads().create_thread(title=title))

    @api.get("/api/threads/{thread_id}")
    def get_thread(thread_id: str) -> dict:
        safe_id = safe_session_id(thread_id)
        thread = threads().get_thread(safe_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return thread_detail_response(thread)

    @api.patch("/api/threads/{thread_id}")
    def update_thread(thread_id: str, request: ThreadUpdateRequest) -> dict:
        safe_id = safe_session_id(thread_id)
        try:
            return thread_detail_response(
                threads().rename_thread(safe_id, request.title or DEFAULT_THREAD_TITLE)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Thread not found.") from None

    @api.delete("/api/threads/{thread_id}")
    def delete_thread(thread_id: str) -> dict:
        safe_id = safe_session_id(thread_id)
        deleted = threads().delete_thread(safe_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found.")
        current_engine = loaded_runtime(safe_id)
        if current_engine is not None:
            if hasattr(current_engine, "clear_loop_session"):
                current_engine.clear_loop_session(safe_id)
            clear_chat_history_for_session(current_engine, safe_id)
        drop_runtime(safe_id)
        return {"deleted": True, "thread_id": safe_id}

    @api.get("/api/threads/{thread_id}/runs")
    def list_thread_runs(thread_id: str) -> dict:
        safe_id = safe_session_id(thread_id)
        if threads().get_thread(safe_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return {
            "thread_id": safe_id,
            "runs": [
                loop_run_summary_response(run)
                for run in threads().list_loop_runs(safe_id)
            ],
        }

    @api.get("/api/threads/{thread_id}/runs/{run_id}")
    def get_thread_run(thread_id: str, run_id: str) -> dict:
        safe_id = safe_session_id(thread_id)
        safe_run_id = safe_path_id(run_id, label="run id")
        run = threads().get_loop_run(safe_id, safe_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Loop run not found.")
        return loop_run_detail_response(run)

    @api.get("/api/recipes")
    def list_recipes() -> dict:
        store = threads()
        recipes = store.list_recipes()
        return {
            "default_recipe_id": DEFAULT_LOOP_RECIPE_ID,
            "recipes": [recipe_summary_response(recipe) for recipe in recipes],
        }

    @api.post("/api/recipes")
    def create_recipe(request: RecipeWriteRequest) -> dict:
        try:
            recipe = threads().create_recipe(
                recipe_id=(
                    safe_path_id(request.recipe_id, label="recipe id")
                    if request.recipe_id
                    else None
                ),
                name=request.name,
                description=request.description or "",
                goal=request.goal,
                instructions=request.instructions or "",
                success_criteria=tuple(request.success_criteria or ()),
                stop_condition=request.stop_condition or "",
                context_provider=request.context_provider or "smart",
                model_profile=request.model_profile or "quality",
                verifier=request.verifier or "default",
                metadata=request.metadata or {},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return recipe_detail_response(recipe)

    @api.get("/api/recipes/{recipe_id}/export")
    def export_recipe(recipe_id: str) -> dict:
        safe_recipe_id = safe_path_id(recipe_id, label="recipe id")
        recipe = threads().get_recipe(safe_recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404, detail="Recipe not found.")
        payload = recipe_detail_response(recipe)
        payload["exported_from"] = APP_TITLE
        return payload

    @api.get("/api/recipes/{recipe_id}")
    def get_recipe(recipe_id: str) -> dict:
        safe_recipe_id = safe_path_id(recipe_id, label="recipe id")
        recipe = threads().get_recipe(safe_recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404, detail="Recipe not found.")
        return recipe_detail_response(recipe)

    @api.patch("/api/recipes/{recipe_id}")
    def update_recipe(recipe_id: str, request: RecipePatchRequest) -> dict:
        safe_recipe_id = safe_path_id(recipe_id, label="recipe id")
        try:
            recipe = threads().update_recipe(
                safe_recipe_id,
                name=request.name,
                description=request.description,
                goal=request.goal,
                instructions=request.instructions,
                success_criteria=(
                    tuple(request.success_criteria)
                    if request.success_criteria is not None
                    else None
                ),
                stop_condition=request.stop_condition,
                context_provider=request.context_provider,
                model_profile=request.model_profile,
                verifier=request.verifier,
                metadata=request.metadata,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Recipe not found.") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return recipe_detail_response(recipe)

    @api.delete("/api/recipes/{recipe_id}")
    def delete_recipe(recipe_id: str) -> dict:
        safe_recipe_id = safe_path_id(recipe_id, label="recipe id")
        deleted = threads().delete_recipe(safe_recipe_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Recipe not found.")
        return {"deleted": True, "recipe_id": safe_recipe_id}

    @api.post("/api/documents")
    async def upload_document(
        file: UploadFile = File(...),
        text_encoding: str = Form("auto"),
        session_id: Optional[str] = Form(None),
    ) -> JSONResponse:
        explicit_session_id = bool(str(session_id or "").strip())
        safe_id = safe_session_id(session_id)
        upload_thread = (
            threads().get_thread(safe_id)
            if explicit_session_id
            else threads().ensure_thread(safe_id)
        )
        if upload_thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        expected_instance_id = upload_thread.instance_id
        uploaded_name = safe_upload_name(file.filename)
        selected_encoding = normalize_text_encoding(text_encoding)
        if selected_encoding is None:
            raise HTTPException(status_code=400, detail="Unsupported text encoding.")

        with TemporaryDirectory(prefix="ai-loop-upload-") as temp_dir:
            upload_path = Path(temp_dir) / uploaded_name
            await write_upload_file(file, upload_path)
            if not thread_instance_is_current(
                safe_id,
                expected_instance_id=expected_instance_id,
            ):
                return stale_upload_response(safe_id)
            current_engine = runtime(safe_id)
            pre_upload_status = current_engine.status()
            try:
                qa_status = current_engine.process_document(
                    str(upload_path),
                    text_encoding=selected_encoding,
                )
                if not thread_instance_is_current(
                    safe_id,
                    expected_instance_id=expected_instance_id,
                ):
                    return stale_upload_response(
                        safe_id,
                        upload_runtime=current_engine,
                    )
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    }
                )
            except DocumentProcessingError as exc:
                LOGGER.warning("Document processing failed: %s", exc)
                if not thread_instance_is_current(
                    safe_id,
                    expected_instance_id=expected_instance_id,
                ):
                    return stale_upload_response(
                        safe_id,
                        upload_runtime=current_engine,
                    )
                qa_status = exc.status
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    },
                    status_code=400,
                )
            except RuntimeError as exc:
                LOGGER.exception("Unexpected document processing failure: %s", exc)
                if not thread_instance_is_current(
                    safe_id,
                    expected_instance_id=expected_instance_id,
                ):
                    return stale_upload_response(
                        safe_id,
                        upload_runtime=current_engine,
                    )
                qa_status = status_with_unexpected_upload_error(
                    pre_upload_status,
                    uploaded_name,
                    selected_encoding,
                    exc,
                )
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    },
                    status_code=500,
                )

    @api.post("/api/query")
    def query(request: QueryRequest) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")
        session_id = safe_session_id(request.session_id)
        store = threads()
        thread = store.ensure_thread(session_id)
        recipe_id = (
            safe_path_id(request.recipe_id, label="recipe id")
            if request.recipe_id
            else DEFAULT_LOOP_RECIPE_ID
        )
        if recipe_id == DEFAULT_LOOP_RECIPE_ID:
            recipe = store.ensure_default_recipe()
        else:
            recipe = store.get_recipe(recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404, detail="Recipe not found.")
        expected_generation = thread.generation
        recent_messages = store.recent_messages(
            session_id,
            limit=MAX_QUERY_HISTORY_MESSAGES,
        )
        conversation_history = conversation_history_from_messages(recent_messages)
        current_runtime = runtime(session_id)
        semantic_memory, semantic_memory_status = semantic_memory_context_for_query(
            current_runtime,
            store,
            session_id=session_id,
            message=message,
            exclude_message_ids=tuple(message.id for message in recent_messages),
        )
        payload = query_response_dict(
            query_result := current_runtime.query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
                loop_recipe=recipe.runtime_dict(),
                context_provider=request.context_provider,
            )
        )
        raw_loop_report = (
            query_result.loop_report.to_dict() if query_result.loop_report else None
        )
        public_loop_report = (
            query_result.loop_report.to_public_dict()
            if query_result.loop_report
            else None
        )
        persisted_turn = store.append_turn(
            session_id,
            user_content=message,
            assistant_content=str(payload.get("answer") or ""),
            thinking=(payload.get("trace") or {}).get("model_thinking"),
            loop_payload=payload,
            raw_loop_report=raw_loop_report,
            public_loop_report=public_loop_report,
            expected_generation=expected_generation,
            expected_instance_id=thread.instance_id,
            title_if_empty=title_from_message(message),
        )
        if persisted_turn:
            index_thread_memory(current_runtime, store, persisted_turn)
            updated_thread = store.get_thread(session_id)
            if updated_thread is not None:
                payload["thread"] = thread_summary_response(updated_thread)
            if raw_loop_report:
                run_id = ((raw_loop_report.get("run") or {}).get("run_id"))
                if run_id:
                    persisted_run = store.get_loop_run(session_id, str(run_id))
                    if persisted_run is not None:
                        payload["run"] = persisted_run.summary_dict()
        payload["recipe"] = recipe.summary_dict()
        return payload

    @api.post("/api/chat/clear")
    def clear_chat(request: Optional[ClearChatRequest] = Body(default=None)) -> dict:
        session_id = safe_session_id(request.session_id if request else None)
        current_engine = loaded_runtime(session_id)
        if current_engine is not None:
            if hasattr(current_engine, "clear_loop_session"):
                current_engine.clear_loop_session(session_id)
            clear_chat_history_for_session(current_engine, session_id)
        threads().clear_thread(session_id)
        return empty_query_response_dict()

    return api


app = create_app()


def main() -> None:
    host = os.getenv("WEB_HOST", os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("PORT", os.getenv("WEB_PORT", "7860")))
    log_level = "debug" if env_flag("APP_DEBUG") else "info"
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
