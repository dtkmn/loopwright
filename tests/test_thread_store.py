import pytest

from src.loop_engine import (
    DEFAULT_LOOP_RECIPE_ID,
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRecipe,
    LoopRun,
    LoopStep,
)
from src.thread_store import DEFAULT_THREAD_TITLE, ThreadStore


def sample_loop_report(*, run_id="run_sample", thread_id="thread_local"):
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id=thread_id,
            user_input="What is Project Phoenix?",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            steps=(
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft direct answer",
                    output_summary="drafted",
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Project Phoenix is a loop workbench.",
            metadata={
                "recipe_id": DEFAULT_LOOP_RECIPE_ID,
                "recipe_name": "General assistant loop",
            },
        )
    )


def test_thread_store_persists_threads_messages_and_latest_payload(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    thread = store.create_thread(title="Phoenix", thread_id="thread_phoenix")
    store.append_message("thread_phoenix", role="user", content="What happened?")
    store.append_turn(
        "thread_phoenix",
        user_content="What is next?",
        assistant_content="Project Phoenix launched.",
        thinking={"available": True, "content": "I checked the loop evidence."},
        loop_payload={"summary": {"final_decision": "not_verified"}},
    )
    store.close()

    restored = ThreadStore(db_path)
    restored_thread = restored.get_thread(thread.id)

    assert restored_thread is not None
    assert restored_thread.title == "Phoenix"
    assert restored_thread.message_count == 3
    assert [message.role for message in restored_thread.messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert restored_thread.messages[2].thinking == {
        "available": True,
        "content": "I checked the loop evidence.",
    }
    assert restored_thread.latest == {
        "summary": {"final_decision": "not_verified"}
    }


def test_thread_store_persists_loop_runs_with_public_report(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    public_report = report.to_public_dict()
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")

    store.append_turn(
        "thread_local",
        user_content="What is Project Phoenix?",
        assistant_content="Project Phoenix is a loop workbench.",
        loop_payload={"summary": {"final_decision": "not_verified"}},
        raw_loop_report=report.to_dict(),
        public_loop_report=public_report,
    )
    store.close()

    restored = ThreadStore(db_path)
    thread = restored.get_thread("thread_local")
    runs = restored.list_loop_runs("thread_local")
    run = restored.get_loop_run("thread_local", "run_sample")

    assert thread.loop_run_count == 1
    assert thread.loop_runs[0].run_id == "run_sample"
    assert len(runs) == 1
    assert run is not None
    assert run.summary_dict()["recipe_id"] == DEFAULT_LOOP_RECIPE_ID
    assert run.detail_dict()["report"] == public_report


def test_thread_store_rejects_raw_loop_report_without_public_report():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    raw_report = report.to_dict()
    raw_report["run"]["user_input"] = "SECRET_USER_INPUT"
    raw_report["run"]["final_answer"] = "SECRET_FINAL_ANSWER"
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="public_loop_report"):
        store.append_turn(
            "thread_local",
            user_content="What is Project Phoenix?",
            assistant_content="Project Phoenix is a loop workbench.",
            raw_loop_report=raw_report,
        )

    thread = store.get_thread("thread_local")
    assert thread.message_count == 0
    assert thread.loop_run_count == 0
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_clear_removes_loop_runs():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content="hello",
        assistant_content="answer",
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )

    store.clear_thread("thread_local")

    assert store.get_thread("thread_local").loop_run_count == 0
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_ensures_and_clears_thread():
    store = ThreadStore.in_memory()

    ensured = store.ensure_thread("thread_local")
    store.append_message("thread_local", role="user", content="hello")
    cleared = store.clear_thread("thread_local")

    assert ensured.title == DEFAULT_THREAD_TITLE
    assert cleared.id == "thread_local"
    assert cleared.message_count == 0
    assert cleared.messages == ()


def test_thread_store_returns_recent_messages_in_original_order():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    for index in range(5):
        role = "user" if index % 2 == 0 else "assistant"
        store.append_message(
            "thread_local",
            role=role,
            content=f"message {index}",
        )

    messages = store.recent_messages("thread_local", limit=3)

    assert [(message.role, message.content) for message in messages] == [
        ("user", "message 2"),
        ("assistant", "message 3"),
        ("user", "message 4"),
    ]


def test_thread_store_retrieves_semantic_memories_by_similarity():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    relevant = store.append_message(
        "thread_local",
        role="user",
        content="Dynamic programming stores answers to subproblems.",
    )
    irrelevant = store.append_message(
        "thread_local",
        role="user",
        content="Banana bread needs ripe bananas.",
    )
    store.upsert_message_embedding(
        relevant,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        irrelevant,
        embedding_model="fake-memory",
        vector=[0.0, 1.0],
    )

    memories = store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
    )

    assert [memory.message_id for memory in memories] == [relevant.id]
    assert memories[0].content == "Dynamic programming stores answers to subproblems."
    assert memories[0].score == 1.0


def test_thread_store_semantic_memories_can_exclude_recent_messages():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    old_message = store.append_message(
        "thread_local",
        role="user",
        content="Old relevant memory.",
    )
    recent_message = store.append_message(
        "thread_local",
        role="assistant",
        content="Recent duplicate memory.",
    )
    store.upsert_message_embedding(
        old_message,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        recent_message,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )

    memories = store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
        exclude_message_ids=(recent_message.id,),
    )

    assert [memory.message_id for memory in memories] == [old_message.id]


def test_thread_store_clear_removes_semantic_memories():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    message = store.append_message(
        "thread_local",
        role="user",
        content="Remember this.",
    )
    store.upsert_message_embedding(
        message,
        embedding_model="fake-memory",
        vector=[1.0],
    )

    store.clear_thread("thread_local")

    assert not store.has_message_embeddings("thread_local", "fake-memory")
    assert store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0],
    ) == ()


def test_thread_store_reports_semantic_memory_count():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    first = store.append_message(
        "thread_local",
        role="user",
        content="Remember this.",
    )
    second = store.append_message(
        "thread_local",
        role="assistant",
        content="I will remember it.",
    )
    store.upsert_message_embedding(
        first,
        embedding_model="fake-memory",
        vector=[1.0],
    )
    store.upsert_message_embedding(
        second,
        embedding_model="fake-memory",
        vector=[1.0],
    )

    detailed = store.get_thread("thread_local")
    listed = store.list_threads()[0]

    assert detailed.memory_count == 2
    assert detailed.detail_dict()["memory_count"] == 2
    assert listed.summary_dict()["memory_count"] == 2


def test_thread_store_clear_missing_thread_does_not_recreate():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    assert store.delete_thread("thread_local") is True

    cleared = store.clear_thread("thread_local")

    assert cleared is None
    assert store.get_thread("thread_local") is None


def test_thread_store_delete_removes_messages_runs_and_memories():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")
    messages = store.append_turn(
        "thread_local",
        user_content="remember this",
        assistant_content="stored answer",
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    for message in messages:
        assert store.upsert_message_embedding(
            message,
            embedding_model="fake-memory",
            vector=[1.0, 0.0],
        )

    assert store.get_thread("thread_local").message_count == 2
    assert store.list_loop_runs("thread_local")
    assert store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
    )

    assert store.delete_thread("thread_local") is True

    assert store.get_thread("thread_local") is None
    assert store.recent_messages("thread_local") == ()
    assert store.list_loop_runs("thread_local") == ()
    assert store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
    ) == ()


def test_thread_store_rejects_invalid_message_role():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="role"):
        store.append_message("thread_local", role="system", content="hidden")


def test_thread_store_append_turn_serialization_failure_leaves_no_partial_message():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")

    with pytest.raises(TypeError):
        store.append_turn(
            "thread_local",
            user_content="hello",
            assistant_content="broken",
            thinking={"bad": object()},
        )

    assert store.get_thread("thread_local").messages == ()
    store.append_message("thread_local", role="user", content="later")
    messages = store.get_thread("thread_local").messages
    assert [(message.role, message.content) for message in messages] == [
        ("user", "later")
    ]


def test_thread_store_append_turn_skips_when_generation_changed():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_local")

    store.clear_thread("thread_local")
    result = store.append_turn(
        "thread_local",
        user_content="stale",
        assistant_content="stale answer",
        expected_generation=thread.generation,
        expected_instance_id=thread.instance_id,
    )

    assert result is None
    assert store.get_thread("thread_local").messages == ()


def test_thread_store_append_turn_skips_recreated_thread_with_same_id():
    store = ThreadStore.in_memory()
    old_thread = store.create_thread(thread_id="thread_aba")
    store.delete_thread("thread_aba")
    new_thread = store.create_thread(thread_id="thread_aba")

    result = store.append_turn(
        "thread_aba",
        user_content="stale",
        assistant_content="stale answer",
        expected_generation=old_thread.generation,
        expected_instance_id=old_thread.instance_id,
    )

    assert old_thread.instance_id != new_thread.instance_id
    assert result is None
    assert store.get_thread("thread_aba").messages == ()


def test_thread_store_append_turn_skips_loop_run_when_generation_changed():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_local")
    report = sample_loop_report(thread_id="thread_local")

    store.clear_thread("thread_local")
    result = store.append_turn(
        "thread_local",
        user_content="stale",
        assistant_content="stale answer",
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
        expected_generation=thread.generation,
        expected_instance_id=thread.instance_id,
    )

    assert result is None
    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_append_turn_requires_complete_guard():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_guard")

    with pytest.raises(ValueError, match="must be supplied together"):
        store.append_turn(
            "thread_guard",
            user_content="stale",
            assistant_content="stale answer",
            expected_generation=thread.generation,
        )


def test_thread_store_persists_loop_recipes(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)

    default_recipe = store.ensure_default_recipe()
    custom = store.create_recipe(
        recipe_id="recipe_weekly_review",
        name="Weekly review",
        description="Summarize the week.",
        goal="Produce a concise weekly review.",
        instructions="Be direct and list risks first.",
        success_criteria=("Risks are listed.", "Next actions are clear."),
        stop_condition="Stop after one accepted summary.",
        context_provider="thread",
        model_profile="quality",
        verifier="human_review",
    )
    store.close()

    restored = ThreadStore(db_path)
    recipes = restored.list_recipes()
    restored_custom = restored.get_recipe("recipe_weekly_review")

    assert recipes[0].recipe_id == DEFAULT_LOOP_RECIPE_ID
    assert default_recipe.recipe_id == DEFAULT_LOOP_RECIPE_ID
    assert restored_custom.recipe_id == custom.recipe_id
    assert restored_custom.name == custom.name
    assert restored_custom.goal == custom.goal
    assert restored_custom.success_criteria == (
        "Risks are listed.",
        "Next actions are clear.",
    )


def test_thread_store_refreshes_builtin_default_recipe():
    store = ThreadStore.in_memory()
    stale_default = LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name="General assistant loop",
        description="Default evidence loop behavior.",
        goal="Answer with indexed context.",
        instructions="Use indexed context when available.",
        success_criteria=("Uses indexed context.",),
        context_provider="auto",
        metadata={"built_in": True},
    )
    with store._lock:
        store._insert_recipe(stale_default)
        store._conn.commit()

    refreshed = store.ensure_default_recipe()

    assert refreshed.recipe_id == DEFAULT_LOOP_RECIPE_ID
    assert refreshed.context_provider == "smart"
    assert "web evidence" in refreshed.instructions
    assert "indexed context" not in refreshed.goal.lower()


def test_thread_store_updates_and_deletes_custom_recipes_only():
    store = ThreadStore.in_memory()
    store.ensure_default_recipe()
    store.create_recipe(
        recipe_id="recipe_custom",
        name="Custom",
        goal="Do a custom loop.",
    )

    updated = store.update_recipe(
        "recipe_custom",
        name="Sharper custom",
        success_criteria=("Passes review.",),
    )
    deleted_default = store.delete_recipe(DEFAULT_LOOP_RECIPE_ID)
    deleted_custom = store.delete_recipe("recipe_custom")

    assert updated.name == "Sharper custom"
    assert updated.success_criteria == ("Passes review.",)
    assert deleted_default is False
    assert deleted_custom is True
    assert store.get_recipe("recipe_custom") is None
    assert store.get_recipe(DEFAULT_LOOP_RECIPE_ID) is not None


def test_thread_store_rejects_duplicate_recipe_ids():
    store = ThreadStore.in_memory()
    store.create_recipe(
        recipe_id="recipe_custom",
        name="Custom",
        goal="Do a custom loop.",
    )

    with pytest.raises(ValueError, match="already exists"):
        store.create_recipe(
            recipe_id="recipe_custom",
            name="Duplicate",
            goal="Should fail.",
        )
