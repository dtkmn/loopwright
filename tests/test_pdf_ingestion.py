from contextlib import nullcontext
from pathlib import Path

import pytest

from src.native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

from pypdf import PdfWriter, apply_configuration
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.ai_loop_engine import AILoopEngine, DocumentProcessingError
from src.document_ingestion import load_documents


def write_text_pdf(path: Path, pages: list[str], *, password: str | None = None):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        if not text:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    if password:
        writer.encrypt(password)
    writer.write(path)


def test_real_pdf_text_extraction_preserves_page_locations(tmp_path):
    path = tmp_path / "three-pages.pdf"
    write_text_pdf(path, ["Cobalt access code is 7319.", "", "Mira owns Cobalt."])

    documents = load_documents(str(path), ".pdf")

    assert [document.page_content.strip() for document in documents] == [
        "Cobalt access code is 7319.",
        "Mira owns Cobalt.",
    ]
    assert [document.metadata for document in documents] == [
        {"source": str(path), "page": 0},
        {"source": str(path), "page": 2},
    ]


@pytest.mark.parametrize(
    "replacement_kind", ["malformed", "encrypted", "blank", "parser_limit"]
)
def test_failed_pdf_replacement_preserves_previous_pdf(
    tmp_path, monkeypatch, replacement_kind
):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "local-hashing-384")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "1024")
    engine = AILoopEngine(llm_backend="mock", embeddings_model="local-hashing-384")
    original = tmp_path / "original.pdf"
    write_text_pdf(original, ["The Cobalt access code is 7319."])
    indexed = engine.process_document(str(original))
    assert indexed.processing_report.success
    question = "What is the Cobalt access code?"
    before = engine.query_with_trace(question, context_provider="document")
    assert "7319" in before.answer
    assert before.trace.citations

    replacement = tmp_path / f"{replacement_kind}.pdf"
    if replacement_kind == "encrypted":
        write_text_pdf(replacement, ["The replacement code is 9999."], password="test")
    elif replacement_kind == "blank":
        write_text_pdf(replacement, [""])
    elif replacement_kind == "parser_limit":
        write_text_pdf(replacement, ["The replacement code is 9999."])
    else:
        replacement.write_bytes(b"%PDF-1.7\nThis is a truncated PDF.\n%%EOF\n")

    parser_configuration = (
        apply_configuration(maximum_declared_stream_length=1)
        if replacement_kind == "parser_limit"
        else nullcontext()
    )
    # A scoped limit exercises the parser's real resource-limit exception with
    # a tiny input and leaves production limits unchanged.
    with parser_configuration, pytest.raises(DocumentProcessingError) as failure:
        engine.process_document(str(replacement))
    if replacement_kind == "parser_limit":
        assert "exceeds maximum allowed length" in str(failure.value)

    status = engine.status()
    assert status.document_name == original.name
    assert status.ready_for_queries
    assert not status.processing_report.success
    assert status.processing_report.phase == "load"
    assert status.processing_report.attempted_document_name == replacement.name
    assert status.processing_report.active_document_name == original.name
    after = engine.query_with_trace(question, context_provider="document")
    assert after.answer == before.answer
    assert after.trace.document_name == original.name
    assert after.trace.citations == before.trace.citations
    assert after.trace.error_message is None
    if replacement_kind == "parser_limit":
        recovered = engine.process_document(str(replacement))
        assert recovered.processing_report.success
        assert recovered.document_name == replacement.name
