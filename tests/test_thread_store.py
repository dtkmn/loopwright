import hashlib
import json
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.answer_loop import SELF_CHECK_REFUSAL_ANSWER
from src.loop_engine import (
    ANSWER_CANDIDATE_SHA256_METADATA_KEY,
    DEFAULT_LOOP_RECIPE_ID,
    EVIDENCE_SET_SHA256_METADATA_KEY,
    EvidenceReference,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRecipe,
    LoopRun,
    LoopStep,
    LoopTerminalReason,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
    answer_candidate_sha256,
    evidence_set_sha256,
)
from src.public_projection import PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
from src.thread_store import (
    DEFAULT_THREAD_TITLE,
    QuarantinedLoopRunError,
    ThreadStore,
)


class _GuardCheckInterleavingCursor:
    def __init__(self, cursor, connection_proxy):
        self._cursor = cursor
        self._connection_proxy = connection_proxy

    def fetchone(self):
        row = self._cursor.fetchone()
        self._connection_proxy.mutation_committed = bool(
            self._connection_proxy.mutate_after_guard_read()
        )
        return row

    def fetchall(self):
        rows = self._cursor.fetchall()
        self._connection_proxy.mutation_committed = bool(
            self._connection_proxy.mutate_after_guard_read()
        )
        return rows

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _GuardCheckInterleavingConnection:
    """Run one competing DB mutation after append_turn reads its CAS guard."""

    def __init__(
        self,
        connection,
        mutate_after_guard_read,
        *,
        sql_markers=("AS message_count", "FROM threads", "WHERE id = ?"),
    ):
        self._connection = connection
        self.mutate_after_guard_read = mutate_after_guard_read
        self.sql_markers = tuple(sql_markers)
        self.mutation_committed = None
        self._interleaved = False

    def execute(self, sql, parameters=()):
        cursor = self._connection.execute(sql, parameters)
        if (
            not self._interleaved
            and all(marker in sql for marker in self.sql_markers)
        ):
            self._interleaved = True
            return _GuardCheckInterleavingCursor(cursor, self)
        return cursor

    def __getattr__(self, name):
        return getattr(self._connection, name)


def sample_loop_report(*, run_id="run_sample", thread_id="thread_local"):
    run_started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    step_started_at = run_started_at + timedelta(milliseconds=10)
    step_ended_at = step_started_at + timedelta(milliseconds=10)
    verification_ended_at = step_ended_at + timedelta(milliseconds=10)
    run_completed_at = verification_ended_at + timedelta(milliseconds=10)
    answer = "Project Phoenix is a loop workbench."
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id=thread_id,
            started_at=run_started_at,
            user_input="What is Project Phoenix?",
            context_provider="none",
            backend="mock",
            model_label="MockLLM",
            completed_at=run_completed_at,
            steps=(
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft direct answer",
                    output_summary="drafted",
                    started_at=step_started_at,
                    ended_at=step_ended_at,
                    backend="mock",
                    model_label="MockLLM",
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
                    },
                ),
                LoopStep(
                    step_id="step_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=step_ended_at,
                    ended_at=verification_ended_at,
                    backend="mock",
                    model_label="MockLLM",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.NOT_VERIFIED,
                    ),
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
                    },
                ),
                LoopStep(
                    step_id="step_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=verification_ended_at,
                    ended_at=run_completed_at,
                    backend="mock",
                    model_label="MockLLM",
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_set_sha256(()),
                    },
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            terminal_reason=LoopTerminalReason.NOT_VERIFIED,
            final_answer=answer,
            metadata={
                "recipe_id": DEFAULT_LOOP_RECIPE_ID,
                "recipe_name": "General assistant loop",
            },
        )
    )


def terminal_loop_report(
    *,
    decision,
    terminal_reason,
    run_id,
    thread_id="thread_terminal",
):
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    completed_at = started_at + timedelta(milliseconds=10)
    phase = LoopPhase.REFUSE if decision == LoopDecision.REFUSE else LoopPhase.INPUT
    retry_budget_exhausted = (
        terminal_reason == LoopTerminalReason.RETRY_BUDGET_EXHAUSTED
    )
    terminal_started_at = started_at
    causal_steps = ()
    if terminal_reason == LoopTerminalReason.VERIFICATION_FAILED:
        terminal_started_at = started_at + timedelta(milliseconds=5)
        causal_steps = (
            LoopStep(
                step_id="step_failed_mechanical_check",
                phase=LoopPhase.MECHANICAL_CHECK,
                decision=LoopDecision.NOT_VERIFIED,
                started_at=started_at,
                ended_at=terminal_started_at,
                output_summary="needs_refusal",
            ),
        )
    elif retry_budget_exhausted:
        draft_ended_at = started_at + timedelta(milliseconds=1)
        format_ended_at = started_at + timedelta(milliseconds=2)
        terminal_started_at = started_at + timedelta(milliseconds=5)
        candidate_digest = answer_candidate_sha256("RAW TERMINAL ANSWER")
        evidence_digest = evidence_set_sha256(())
        causal_steps = (
            LoopStep(
                step_id="step_denied_retry_draft",
                phase=LoopPhase.DRAFT,
                decision=LoopDecision.CONTINUE,
                started_at=started_at,
                ended_at=draft_ended_at,
                metadata={
                    ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                    EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                },
            ),
            LoopStep(
                step_id="step_denied_retry_format",
                phase=LoopPhase.FORMAT_CHECK,
                decision=LoopDecision.CONTINUE,
                started_at=draft_ended_at,
                ended_at=format_ended_at,
                output_summary="format_passed",
                metadata={
                    ANSWER_CANDIDATE_SHA256_METADATA_KEY: candidate_digest,
                    EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                },
            ),
            LoopStep(
                step_id="step_denied_mechanical_retry",
                phase=LoopPhase.MECHANICAL_CHECK,
                decision=LoopDecision.NOT_VERIFIED,
                started_at=format_ended_at,
                ended_at=terminal_started_at,
                output_summary="needs_retry",
                metadata={
                    "reasons": ["missing_inline_citation"],
                    "retry_denied": True,
                    "retry_unavailable": False,
                    "retry_budget_exhausted": True,
                },
            ),
        )
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id=thread_id,
            started_at=started_at,
            completed_at=completed_at,
            user_input="Private terminal request",
            context_provider="document",
            backend="mock",
            model_label="MockLLM",
            policy=LoopPolicy(max_retries=0) if retry_budget_exhausted else LoopPolicy(),
            steps=(
                *causal_steps,
                LoopStep(
                    step_id=f"step_{decision.value}",
                    phase=phase,
                    decision=decision,
                    started_at=terminal_started_at,
                    ended_at=completed_at,
                ),
            ),
            final_decision=decision,
            terminal_reason=terminal_reason,
            final_answer="RAW TERMINAL ANSWER",
        )
    )


def legacy_unbound_supported_report(
    *,
    run_id="run_legacy_supported",
    thread_id="thread_local",
):
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    drafted_at = started_at + timedelta(milliseconds=10)
    verified_at = drafted_at + timedelta(milliseconds=10)
    completed_at = verified_at + timedelta(milliseconds=10)
    answer = "Project Phoenix launches in June 2026 [1]."
    evidence = (
        EvidenceReference.from_source(
            citation_id=1,
            provider="document",
            source_identity="project_phoenix.md",
            page=None,
            chunk_index=0,
            excerpt="Project Phoenix launches in June 2026.",
        ),
    )
    evidence_digest = evidence_set_sha256(evidence)
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id=thread_id,
            started_at=started_at,
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="ollama",
            model_label="Ollama",
            completed_at=completed_at,
            steps=(
                LoopStep(
                    step_id="step_legacy_draft",
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    started_at=started_at,
                    ended_at=drafted_at,
                    output_summary=answer,
                    metadata={EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest},
                ),
                LoopStep(
                    step_id="step_legacy_verify",
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    started_at=drafted_at,
                    ended_at=verified_at,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        verifier_backend="ollama",
                    ),
                    metadata={EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest},
                ),
                LoopStep(
                    step_id="step_legacy_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.SUPPORTED,
                    started_at=verified_at,
                    ended_at=completed_at,
                    output_summary=answer,
                    metadata={
                        ANSWER_CANDIDATE_SHA256_METADATA_KEY: (
                            answer_candidate_sha256(answer)
                        ),
                        EVIDENCE_SET_SHA256_METADATA_KEY: evidence_digest,
                    },
                ),
            ),
            evidence=evidence,
            final_decision=LoopDecision.SUPPORTED,
            terminal_reason=LoopTerminalReason.COMPLETED,
            final_answer=answer,
        )
    )


def test_thread_store_does_not_resurface_legacy_loop_payloads(tmp_path):
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
    assert restored_thread.messages[2].thinking is None
    assert restored_thread.latest is None
    assert all(message.loop_payload is None for message in restored_thread.messages)


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
    summary = run.summary_dict()
    assert summary["projection_status"] == "available"
    assert summary["projection_schema_version"] == (
        PUBLIC_REPORT_PROJECTION_SCHEMA_VERSION
    )
    assert summary["terminal_reason"] == "not_verified"
    assert "recipe_id" not in summary
    assert "recipe_name" not in summary
    assert run.detail_dict()["report"] == public_report


def test_thread_store_backfills_binding_witness_once_for_legacy_database(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    with store._lock:
        store._conn.execute("DROP TABLE loop_run_message_bindings")
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    record = restored.get_loop_run("thread_local", report.run.run_id)

    assert record is not None
    assert record.quarantine_reason is None
    binding = restored._conn.execute(
        "SELECT run_id FROM loop_run_message_bindings WHERE run_id = ?",
        (report.run.run_id,),
    ).fetchone()
    assert binding is not None


def test_thread_store_get_thread_uses_one_cross_connection_snapshot(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    reader = ThreadStore(db_path)
    competing = ThreadStore(db_path)
    old_thread = reader.create_thread(
        thread_id="thread_snapshot",
        title="Old instance",
    )
    reader.append_message(
        old_thread.id,
        role="user",
        content="Old instance message",
    )

    def replace_after_thread_row_read():
        assert competing.delete_thread(old_thread.id) is True
        competing.create_thread(
            thread_id=old_thread.id,
            title="Replacement instance",
        )
        competing.append_message(
            old_thread.id,
            role="user",
            content="Replacement instance message",
        )
        return True

    proxy = _GuardCheckInterleavingConnection(
        reader._conn,
        replace_after_thread_row_read,
        sql_markers=("FROM threads t", "WHERE t.id = ?"),
    )
    reader._conn = proxy

    snapshot = reader.get_thread(old_thread.id)
    current = competing.get_thread(old_thread.id)

    assert proxy.mutation_committed is True
    assert snapshot is not None
    assert snapshot.instance_id == old_thread.instance_id
    assert snapshot.title == "Old instance"
    assert [message.content for message in snapshot.messages] == [
        "Old instance message"
    ]
    assert current is not None
    assert current.instance_id != old_thread.instance_id
    assert current.title == "Replacement instance"
    assert [message.content for message in current.messages] == [
        "Replacement instance message"
    ]


def test_thread_store_list_threads_uses_one_cross_connection_snapshot(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    reader = ThreadStore(db_path)
    competing = ThreadStore(db_path)
    old_thread = reader.create_thread(
        thread_id="thread_list_snapshot",
        title="Old instance",
    )
    reader.append_message(
        old_thread.id,
        role="user",
        content="Old instance message",
    )
    old_thread = reader.get_thread(old_thread.id)
    assert old_thread is not None
    replacement_holder = {}

    def replace_after_list_rows_read():
        assert competing.delete_thread(old_thread.id) is True
        replacement = competing.create_thread(
            thread_id=old_thread.id,
            title="Replacement instance",
        )
        first = competing.append_message(
            replacement.id,
            role="user",
            content="Replacement user message",
        )
        second = competing.append_message(
            replacement.id,
            role="assistant",
            content="Replacement assistant message",
        )
        for message in (first, second):
            assert competing.upsert_message_embedding(
                message,
                embedding_model="snapshot-memory",
                vector=[1.0, 0.0],
            )
        replacement_holder["thread"] = competing.get_thread(replacement.id)
        return True

    proxy = _GuardCheckInterleavingConnection(
        reader._conn,
        replace_after_list_rows_read,
        sql_markers=(
            "FROM threads t",
            "LEFT JOIN messages m",
            "ORDER BY t.updated_at DESC",
        ),
    )
    reader._conn = proxy

    listed = next(
        thread
        for thread in reader.list_threads()
        if thread.id == old_thread.id
    )

    assert proxy.mutation_committed is True
    replacement = replacement_holder["thread"]
    assert replacement is not None
    observed_snapshot = (
        listed.title,
        listed.instance_id,
        listed.generation,
        listed.created_at,
        listed.updated_at,
        listed.message_count,
        listed.memory_count,
    )
    old_snapshot = (
        old_thread.title,
        old_thread.instance_id,
        old_thread.generation,
        old_thread.created_at,
        old_thread.updated_at,
        old_thread.message_count,
        old_thread.memory_count,
    )
    replacement_snapshot = (
        replacement.title,
        replacement.instance_id,
        replacement.generation,
        replacement.created_at,
        replacement.updated_at,
        replacement.message_count,
        replacement.memory_count,
    )
    assert observed_snapshot in (old_snapshot, replacement_snapshot)


def test_thread_store_rejects_user_message_report_mismatch_transactionally():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="user_content does not match"):
        store.append_turn(
            "thread_local",
            user_content="A different persisted question",
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_rejects_visible_assistant_report_mismatch_transactionally():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="assistant_content does not match"):
        store.append_turn(
            "thread_local",
            user_content=report.run.user_input,
            assistant_content="A different displayed answer",
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


@pytest.mark.parametrize(
    ("field_name", "invalid_content"),
    (
        ("user_content", None),
        ("user_content", False),
        ("user_content", 0),
        ("user_content", []),
        ("assistant_content", None),
        ("assistant_content", False),
        ("assistant_content", 0),
        ("assistant_content", []),
    ),
)
def test_thread_store_rejects_non_string_bound_turn_content(
    field_name,
    invalid_content,
):
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")
    turn = {
        "user_content": report.run.user_input,
        "assistant_content": report.run.final_answer,
        "raw_loop_report": report.to_dict(),
        "public_loop_report": report.to_public_dict(),
    }
    turn[field_name] = invalid_content

    with pytest.raises(TypeError, match=field_name):
        store.append_turn("thread_local", **turn)

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


@pytest.mark.parametrize(
    ("decision", "terminal_reason", "expected_answer", "wrong_answer"),
    (
        (
            LoopDecision.BLOCK,
            LoopTerminalReason.BLOCKED,
            PUBLIC_REDACTION_TEXT,
            SELF_CHECK_REFUSAL_ANSWER,
        ),
        (
            LoopDecision.REFUSE,
            LoopTerminalReason.VERIFICATION_FAILED,
            SELF_CHECK_REFUSAL_ANSWER,
            PUBLIC_REDACTION_TEXT,
        ),
        (
            LoopDecision.REFUSE,
            LoopTerminalReason.RETRY_BUDGET_EXHAUSTED,
            SELF_CHECK_REFUSAL_ANSWER,
            PUBLIC_REDACTION_TEXT,
        ),
    ),
)
def test_thread_store_binds_existing_terminal_display_mappings_exactly(
    decision,
    terminal_reason,
    expected_answer,
    wrong_answer,
):
    accepted_store = ThreadStore.in_memory()
    accepted_report = terminal_loop_report(
        decision=decision,
        terminal_reason=terminal_reason,
        run_id=f"run_{decision.value}_accepted",
    )
    accepted_store.create_thread(thread_id="thread_terminal")

    accepted_store.append_turn(
        "thread_terminal",
        user_content=accepted_report.run.user_input,
        assistant_content=expected_answer,
        raw_loop_report=accepted_report.to_dict(),
        public_loop_report=accepted_report.to_public_dict(),
    )

    assert accepted_store.get_thread("thread_terminal").messages[-1].content == (
        expected_answer
    )

    rejected_store = ThreadStore.in_memory()
    rejected_report = terminal_loop_report(
        decision=decision,
        terminal_reason=terminal_reason,
        run_id=f"run_{decision.value}_rejected",
    )
    rejected_store.create_thread(thread_id="thread_terminal")

    with pytest.raises(ValueError, match="assistant_content does not match"):
        rejected_store.append_turn(
            "thread_terminal",
            user_content=rejected_report.run.user_input,
            assistant_content=wrong_answer,
            raw_loop_report=rejected_report.to_dict(),
            public_loop_report=rejected_report.to_public_dict(),
        )

    assert rejected_store.get_thread("thread_terminal").messages == ()
    assert rejected_store.list_loop_runs("thread_terminal") == ()


def test_thread_store_forces_canonical_terminal_thinking_redaction():
    store = ThreadStore.in_memory()
    report = terminal_loop_report(
        decision=LoopDecision.BLOCK,
        terminal_reason=LoopTerminalReason.BLOCKED,
        run_id="run_terminal_thinking",
    )
    secret = "SECRET_CALLER_SUPPLIED_TERMINAL_THINKING"
    store.create_thread(thread_id="thread_terminal")

    persisted = store.append_turn(
        "thread_terminal",
        user_content=report.run.user_input,
        assistant_content=PUBLIC_REDACTION_TEXT,
        thinking={
            "available": True,
            "redacted": False,
            "content": secret,
            "extra": secret,
        },
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )

    assert persisted is not None
    assert persisted[-1].thinking == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    stored_json = store._conn.execute(
        "SELECT thinking_json FROM messages WHERE id = ?",
        (persisted[-1].id,),
    ).fetchone()["thinking_json"]
    assert secret not in stored_json


def test_thread_store_binds_nonterminal_thinking_to_raw_run_digest_or_drops_it():
    bound_thinking = "Bound model thinking"
    forged_thinking = "FORGED_UNBOUND_MODEL_THINKING"
    base_report = sample_loop_report(thread_id="thread_local")
    report = replace(
        base_report,
        run=replace(
            base_report.run,
            metadata={
                **base_report.run.metadata,
                "model_thinking_sha256": hashlib.sha256(
                    bound_thinking.encode("utf-8")
                ).hexdigest(),
            },
        ),
    )

    accepted_store = ThreadStore.in_memory()
    accepted_store.create_thread(thread_id="thread_local")
    accepted = accepted_store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        thinking={"content": bound_thinking, "extra": "must be dropped"},
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert accepted is not None
    assert accepted[-1].thinking == {
        "available": True,
        "redacted": False,
        "label": "Model Thinking (unverified)",
        "content": bound_thinking,
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    accepted_json = accepted_store._conn.execute(
        "SELECT thinking_json FROM messages WHERE id = ?",
        (accepted[-1].id,),
    ).fetchone()["thinking_json"]
    assert "must be dropped" not in accepted_json

    rejected_store = ThreadStore.in_memory()
    rejected_store.create_thread(thread_id="thread_local")
    rejected = rejected_store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        thinking={"content": forged_thinking},
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert rejected is not None
    assert rejected[-1].thinking is None
    stored_json = rejected_store._conn.execute(
        "SELECT thinking_json FROM messages WHERE id = ?",
        (rejected[-1].id,),
    ).fetchone()["thinking_json"]
    assert stored_json is None


def test_thread_store_rebinds_terminal_thinking_on_restart(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    report = terminal_loop_report(
        decision=LoopDecision.BLOCK,
        terminal_reason=LoopTerminalReason.BLOCKED,
        run_id="run_terminal_restart_thinking",
    )
    secret = "SECRET_LEGACY_TERMINAL_THINKING"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_terminal")
    persisted = store.append_turn(
        "thread_terminal",
        user_content=report.run.user_input,
        assistant_content=PUBLIC_REDACTION_TEXT,
        thinking={"content": "original terminal thinking"},
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert persisted is not None
    with store._lock:
        store._conn.execute(
            "UPDATE messages SET thinking_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "available": True,
                        "redacted": False,
                        "label": "Model Thinking (unverified)",
                        "content": secret,
                        "note": secret,
                    }
                ),
                persisted[-1].id,
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    thread = restored.get_thread("thread_terminal")

    assert thread is not None
    assert thread.messages[-1].thinking == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    assert secret not in json.dumps(thread.detail_dict())


@pytest.mark.parametrize("corruption", ("thinking_digest", "missing_binding"))
def test_thread_store_suppresses_unbound_nonterminal_thinking_on_restart(
    tmp_path,
    corruption,
):
    db_path = tmp_path / f"threads-{corruption}.sqlite3"
    bound_thinking = "Thinking bound to the canonical run."
    forged_thinking = "SECRET_UNBOUND_NONTERMINAL_THINKING"
    base_report = sample_loop_report(thread_id="thread_local")
    report = replace(
        base_report,
        run=replace(
            base_report.run,
            metadata={
                **base_report.run.metadata,
                "model_thinking_sha256": hashlib.sha256(
                    bound_thinking.encode("utf-8")
                ).hexdigest(),
            },
        ),
    )
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    persisted = store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        thinking={"content": bound_thinking},
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert persisted is not None
    assert store.get_thread("thread_local").messages[-1].thinking["content"] == (
        bound_thinking
    )
    with store._lock:
        if corruption == "thinking_digest":
            store._conn.execute(
                "UPDATE messages SET thinking_json = ? WHERE id = ?",
                (
                    json.dumps({"content": forged_thinking}),
                    persisted[-1].id,
                ),
            )
        else:
            store._conn.execute(
                "DELETE FROM loop_run_message_bindings WHERE run_id = ?",
                (report.run.run_id,),
            )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    thread = restored.get_thread("thread_local")

    assert thread is not None
    assert forged_thinking not in json.dumps(thread.detail_dict())
    if corruption == "missing_binding":
        assert thread.messages == ()
        assert thread.loop_runs[0].quarantine_reason == (
            "stored_message_binding_invalid"
        )
    else:
        assert thread.messages[-1].thinking is None


def test_thread_store_rejects_raw_loop_report_without_public_report():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    raw_report = report.to_dict()
    raw_report["run"]["user_input"] = "SECRET_USER_INPUT"
    raw_report["run"]["final_answer"] = "SECRET_FINAL_ANSWER"
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="must be supplied together"):
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


def test_thread_store_rejects_public_loop_report_without_raw_report():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="must be supplied together"):
        store.append_turn(
            "thread_local",
            user_content="What is Project Phoenix?",
            assistant_content="Project Phoenix is a loop workbench.",
            public_loop_report=report.to_public_dict(),
        )

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_rejects_cross_run_public_projection_transactionally():
    store = ThreadStore.in_memory()
    raw_report = sample_loop_report(
        run_id="run_alpha",
        thread_id="thread_alpha",
    )
    foreign_public_report = sample_loop_report(
        run_id="run_beta",
        thread_id="thread_beta",
    ).to_public_dict()
    foreign_public_report["run"]["final_answer"] = "FOREIGN_PUBLIC_PAYLOAD"
    store.create_thread(thread_id="thread_alpha")

    with pytest.raises(ValueError, match="canonical public projection"):
        store.append_turn(
            "thread_alpha",
            user_content=raw_report.run.user_input,
            assistant_content=raw_report.run.final_answer,
            raw_loop_report=raw_report.to_dict(),
            public_loop_report=foreign_public_report,
        )

    assert store.get_thread("thread_alpha").messages == ()
    assert store.list_loop_runs("thread_alpha") == ()


def test_thread_store_rejects_raw_session_identity_mismatch_transactionally():
    store = ThreadStore.in_memory()
    foreign_report = sample_loop_report(
        run_id="run_alpha",
        thread_id="thread_beta",
    )
    store.create_thread(thread_id="thread_alpha")

    with pytest.raises(ValueError, match="session identity"):
        store.append_turn(
            "thread_alpha",
            user_content="alpha",
            assistant_content="alpha answer",
            raw_loop_report=foreign_report.to_dict(),
            public_loop_report=foreign_report.to_public_dict(),
        )

    assert store.get_thread("thread_alpha").messages == ()
    assert store.list_loop_runs("thread_alpha") == ()


@pytest.mark.parametrize(
    "poison_field",
    ("backend_object", "answer_object", "step_model_object", "retry_infinity"),
)
def test_thread_store_rejects_noncanonical_raw_field_types_transactionally(
    poison_field,
):
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    raw_report = report.to_dict()
    if poison_field == "backend_object":
        raw_report["run"]["backend"] = {"secret": "RAW_BACKEND_SECRET"}
    elif poison_field == "answer_object":
        raw_report["run"]["final_answer"] = {"secret": "RAW_ANSWER_SECRET"}
    elif poison_field == "step_model_object":
        raw_report["run"]["steps"][0]["model_label"] = {
            "secret": "RAW_STEP_SECRET"
        }
    else:
        raw_report["run"]["steps"][0]["retry_count"] = float("inf")
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError):
        store.append_turn(
            "thread_local",
            user_content="question",
            assistant_content="answer",
            raw_loop_report=raw_report,
            public_loop_report=report.to_public_dict(),
        )

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


def test_thread_store_rejects_noncanonical_run_identity_transactionally():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    raw_report = report.to_dict()
    public_report = report.to_public_dict()
    raw_report["run"]["run_id"] = " run_sample "
    public_report["run"]["run_id"] = " run_sample "
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="canonical string"):
        store.append_turn(
            "thread_local",
            user_content="question",
            assistant_content="answer",
            raw_loop_report=raw_report,
            public_loop_report=public_report,
        )

    assert store.get_thread("thread_local").messages == ()
    assert store.list_loop_runs("thread_local") == ()


@pytest.mark.parametrize(
    ("run_id", "thread_id"),
    (
        ("r" * 97, "thread_local"),
        ("run_\u00e9", "thread_local"),
        ("run_local", "t" * 97),
        ("run_local", "thread_\u00e9"),
    ),
)
def test_thread_store_rejects_identities_the_api_cannot_address(
    run_id,
    thread_id,
):
    store = ThreadStore.in_memory()
    report = sample_loop_report(run_id=run_id, thread_id=thread_id)

    with pytest.raises(ValueError):
        store.append_turn(
            thread_id,
            user_content="question",
            assistant_content="answer",
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    assert store.list_threads() == []


def test_thread_store_failed_run_insert_rolls_back_auto_created_thread():
    store = ThreadStore.in_memory()
    first_report = sample_loop_report(
        run_id="run_duplicate",
        thread_id="thread_alpha",
    )
    store.append_turn(
        "thread_alpha",
        user_content=first_report.run.user_input,
        assistant_content=first_report.run.final_answer,
        raw_loop_report=first_report.to_dict(),
        public_loop_report=first_report.to_public_dict(),
    )
    duplicate_report = sample_loop_report(
        run_id="run_duplicate",
        thread_id="thread_beta",
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.append_turn(
            "thread_beta",
            user_content=duplicate_report.run.user_input,
            assistant_content=duplicate_report.run.final_answer,
            raw_loop_report=duplicate_report.to_dict(),
            public_loop_report=duplicate_report.to_public_dict(),
        )

    assert store.get_thread("thread_beta") is None
    assert store.get_thread("thread_alpha").message_count == 2


@pytest.mark.parametrize(
    "poisoned_public_json",
    (
        "not-json",
        '{"run":"legacy-shape"}',
        json.dumps(
            {
                "projection_schema_version": "loop-public-report/v0",
                "run": {
                    "run_id": "run_foreign",
                    "session_id": "thread_foreign",
                    "final_answer": "FOREIGN_PUBLIC_PAYLOAD",
                },
            }
        ),
    ),
)
def test_thread_store_reprojects_valid_raw_and_ignores_poisoned_public_cache(
    tmp_path,
    poisoned_public_json,
):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    with store._lock:
        store._conn.execute(
            "UPDATE loop_runs SET public_report_json = ? WHERE run_id = ?",
            (poisoned_public_json, "run_sample"),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    record = restored.get_loop_run("thread_local", "run_sample")

    assert record is not None
    assert record.quarantine_reason is None
    assert record.public_report == report.to_public_dict()
    assert record.detail_dict()["report"]["run"]["run_id"] == "run_sample"
    assert "FOREIGN_PUBLIC_PAYLOAD" not in json.dumps(record.detail_dict())


@pytest.mark.parametrize(
    "raw_mutator",
    (
        lambda raw: {"run": "legacy-shape"},
        lambda raw: {
            **raw,
            "run": {**raw["run"], "run_id": "run_foreign"},
        },
        lambda raw: {
            **raw,
            "run": {**raw["run"], "session_id": "thread_foreign"},
        },
        lambda raw: {**raw, "schema_version": "loop-report/v0"},
        lambda raw: {
            **raw,
            "run": {
                **raw["run"],
                "backend": {"secret": "RAW_BACKEND_SECRET"},
            },
        },
        lambda raw: {
            **raw,
            "run": {
                **raw["run"],
                "final_answer": {"secret": "RAW_ANSWER_SECRET"},
            },
        },
        lambda raw: {
            **raw,
            "run": {
                **raw["run"],
                "steps": [
                    {
                        **raw["run"]["steps"][0],
                        "model_label": {"secret": "RAW_STEP_SECRET"},
                    }
                ],
            },
        },
        lambda raw: {
            **raw,
            "run": {
                **raw["run"],
                "steps": [
                    {
                        **raw["run"]["steps"][0],
                        "retry_count": float("inf"),
                    }
                ],
            },
        },
    ),
)
def test_thread_store_quarantines_invalid_or_mismatched_raw_rows(
    tmp_path,
    raw_mutator,
):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    poisoned_raw = raw_mutator(report.to_dict())
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_runs
            SET raw_report_json = ?, public_report_json = ?
            WHERE run_id = ?
            """,
            (
                json.dumps(poisoned_raw),
                json.dumps({"run": {"final_answer": "CACHED_PUBLIC_SECRET"}}),
                "run_sample",
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    thread = restored.get_thread("thread_local")
    listed = restored.list_loop_runs("thread_local")
    record = restored.get_loop_run("thread_local", "run_sample")

    assert thread is not None
    assert len(thread.loop_runs) == 1
    assert len(listed) == 1
    assert record is not None
    assert record.summary_dict() == {
        "run_id": "run_sample",
        "thread_id": "thread_local",
        "created_at": record.created_at,
        "projection_status": "quarantined",
        "quarantine_reason": "stored_loop_report_invalid",
    }
    assert "CACHED_PUBLIC_SECRET" not in json.dumps(record.summary_dict())
    with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
        record.detail_dict()


def test_thread_store_quarantines_duplicate_raw_json_keys(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    raw_json = json.dumps(report.to_dict(), separators=(",", ":"))
    canonical_identity = '"run_id":"run_sample"'
    assert raw_json.count(canonical_identity) == 1
    duplicate_identity_json = raw_json.replace(
        canonical_identity,
        '"run_id":"run_foreign","run_id":"run_sample"',
        1,
    )
    with store._lock:
        store._conn.execute(
            "UPDATE loop_runs SET raw_report_json = ? WHERE run_id = ?",
            (duplicate_identity_json, "run_sample"),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    record = restored.get_loop_run("thread_local", "run_sample")

    assert record is not None
    assert record.quarantine_reason == "stored_loop_report_invalid"
    assert record.summary_dict()["projection_status"] == "quarantined"
    with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
        record.detail_dict()


def test_thread_store_rejects_nonfinite_message_json_before_write():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="JSON compliant"):
        store.append_message(
            "thread_local",
            role="assistant",
            content="answer",
            thinking={"score": float("nan")},
        )

    assert store.get_thread("thread_local").messages == ()


@pytest.mark.parametrize(
    "poisoned_thinking_json",
    (
        '{"available":true,"available":false}',
        '{"score":NaN}',
        '["not-an-object"]',
    ),
)
def test_thread_store_suppresses_malformed_optional_thinking_on_restart(
    tmp_path,
    poisoned_thinking_json,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    message = store.append_message(
        "thread_local",
        role="assistant",
        content="Persisted answer remains readable.",
        thinking={"available": True, "content": "Valid before corruption."},
    )
    with store._lock:
        store._conn.execute(
            "UPDATE messages SET thinking_json = ? WHERE id = ?",
            (poisoned_thinking_json, message.id),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    thread = restored.get_thread("thread_local")

    assert thread is not None
    assert thread.messages[0].content == "Persisted answer remains readable."
    assert thread.messages[0].thinking is None


@pytest.mark.parametrize(
    "corruption",
    (
        "user_content",
        "assistant_content",
        "missing_user",
        "cross_thread_user",
        "wrong_role_user",
    ),
)
def test_thread_store_quarantines_invalid_message_binding_on_restart(
    tmp_path,
    corruption,
):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_local")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    persisted_turn = store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert persisted_turn is not None
    user_message, assistant_message = persisted_turn

    with store._lock:
        if corruption == "user_content":
            store._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("FORGED USER MESSAGE", user_message.id),
            )
        elif corruption == "assistant_content":
            store._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("FORGED ASSISTANT MESSAGE", assistant_message.id),
            )
        elif corruption == "missing_user":
            store._conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (user_message.id,),
            )
        elif corruption == "cross_thread_user":
            store.create_thread(thread_id="thread_foreign")
            foreign_message = store.append_message(
                "thread_foreign",
                role="user",
                content=report.run.user_input,
            )
            store._conn.execute(
                "UPDATE loop_runs SET user_message_id = ? WHERE run_id = ?",
                (foreign_message.id, report.run.run_id),
            )
        else:
            wrong_role_message = store.append_message(
                "thread_local",
                role="assistant",
                content=report.run.user_input,
            )
            store._conn.execute(
                "UPDATE loop_runs SET user_message_id = ? WHERE run_id = ?",
                (wrong_role_message.id, report.run.run_id),
            )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    direct_record = restored.get_loop_run("thread_local", report.run.run_id)
    listed_record = restored.list_loop_runs("thread_local")[0]
    thread_record = restored.get_thread("thread_local").loop_runs[0]

    for record in (direct_record, listed_record, thread_record):
        assert record is not None
        assert record.quarantine_reason == "stored_message_binding_invalid"
        assert record.summary_dict()["projection_status"] == "quarantined"
        with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
            record.detail_dict()


@pytest.mark.parametrize(
    "corruption",
    (
        "raw_report",
        "missing_binding",
        "user_content",
        "assistant_content",
    ),
)
def test_thread_store_hides_entire_invalid_linked_pair_from_message_reads(
    tmp_path,
    corruption,
):
    db_path = tmp_path / "threads.sqlite3"
    report = sample_loop_report(thread_id="thread_pair_boundary")
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_pair_boundary")
    persisted_turn = store.append_turn(
        "thread_pair_boundary",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert persisted_turn is not None
    user_message, assistant_message = persisted_turn
    legacy_message = store.append_message(
        "thread_pair_boundary",
        role="assistant",
        content="Unbound legacy message remains visible.",
    )
    for message in persisted_turn:
        assert store.upsert_message_embedding(
            message,
            embedding_model="invalid-linked-memory",
            vector=[1.0, 0.0],
        )
    assert store.upsert_message_embedding(
        legacy_message,
        embedding_model="legacy-memory",
        vector=[1.0, 0.0],
    )

    with store._lock:
        if corruption == "raw_report":
            store._conn.execute(
                "UPDATE loop_runs SET raw_report_json = ? WHERE run_id = ?",
                ('{"forged":true}', report.run.run_id),
            )
        elif corruption == "missing_binding":
            store._conn.execute(
                "DELETE FROM loop_run_message_bindings WHERE run_id = ?",
                (report.run.run_id,),
            )
        elif corruption == "user_content":
            store._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("FORGED USER MESSAGE", user_message.id),
            )
        else:
            store._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("FORGED ASSISTANT MESSAGE", assistant_message.id),
            )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    detailed = restored.get_thread("thread_pair_boundary")
    recent = restored.recent_messages("thread_pair_boundary")
    listed = next(
        thread
        for thread in restored.list_threads()
        if thread.id == "thread_pair_boundary"
    )

    assert detailed is not None
    assert [(message.id, message.content) for message in detailed.messages] == [
        (legacy_message.id, "Unbound legacy message remains visible.")
    ]
    assert [(message.id, message.content) for message in recent] == [
        (legacy_message.id, "Unbound legacy message remains visible.")
    ]
    assert restored.semantic_memories(
        "thread_pair_boundary",
        embedding_model="invalid-linked-memory",
        query_vector=[1.0, 0.0],
    ) == ()
    assert [
        memory.message_id
        for memory in restored.semantic_memories(
            "thread_pair_boundary",
            embedding_model="legacy-memory",
            query_vector=[1.0, 0.0],
        )
    ] == [legacy_message.id]
    assert detailed.message_count == 1
    assert detailed.memory_count == 1
    assert listed.message_count == 1
    assert listed.memory_count == 1
    assert not restored.has_message_embeddings(
        "thread_pair_boundary",
        "invalid-linked-memory",
    )
    assert restored.has_message_embeddings(
        "thread_pair_boundary",
        "legacy-memory",
    )


def test_thread_store_reads_valid_linked_pair_from_canonical_report():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_valid_pair")
    store.create_thread(thread_id="thread_valid_pair")
    persisted_turn = store.append_turn(
        "thread_valid_pair",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
        raw_loop_report=report.to_dict(),
        public_loop_report=report.to_public_dict(),
    )
    assert persisted_turn is not None
    for message in persisted_turn:
        assert store.upsert_message_embedding(
            message,
            embedding_model="valid-linked-memory",
            vector=[1.0, 0.0],
        )

    detailed = store.get_thread("thread_valid_pair")
    recent = store.recent_messages("thread_valid_pair")
    listed = next(
        thread
        for thread in store.list_threads()
        if thread.id == "thread_valid_pair"
    )
    expected_contents = [
        report.run.user_input,
        report.to_public_dict()["run"]["final_answer"],
    ]

    assert detailed is not None
    assert [message.content for message in detailed.messages] == expected_contents
    assert [message.content for message in recent] == expected_contents
    assert [
        memory.content
        for memory in store.semantic_memories(
            "thread_valid_pair",
            embedding_model="valid-linked-memory",
            query_vector=[1.0, 0.0],
        )
    ] == expected_contents
    assert detailed.message_count == 2
    assert detailed.memory_count == 2
    assert listed.message_count == 2
    assert listed.memory_count == 2
    assert store.has_message_embeddings(
        "thread_valid_pair",
        "valid-linked-memory",
    )


def test_thread_store_keeps_genuinely_unbound_legacy_messages_readable():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_unbound_legacy")
    user_message = store.append_message(
        "thread_unbound_legacy",
        role="user",
        content="Legacy user message.",
    )
    assistant_message = store.append_message(
        "thread_unbound_legacy",
        role="assistant",
        content="Legacy assistant message.",
    )
    for message in (user_message, assistant_message):
        assert store.upsert_message_embedding(
            message,
            embedding_model="legacy-only-memory",
            vector=[1.0, 0.0],
        )

    detailed = store.get_thread("thread_unbound_legacy")
    recent = store.recent_messages("thread_unbound_legacy")
    listed = next(
        thread
        for thread in store.list_threads()
        if thread.id == "thread_unbound_legacy"
    )
    expected_contents = ["Legacy user message.", "Legacy assistant message."]

    assert detailed is not None
    assert [message.content for message in detailed.messages] == expected_contents
    assert [message.content for message in recent] == expected_contents
    assert [
        memory.content
        for memory in store.semantic_memories(
            "thread_unbound_legacy",
            embedding_model="legacy-only-memory",
            query_vector=[1.0, 0.0],
        )
    ] == expected_contents
    assert detailed.message_count == listed.message_count == 2
    assert detailed.memory_count == listed.memory_count == 2
    assert store.has_message_embeddings(
        "thread_unbound_legacy",
        "legacy-only-memory",
    )


def test_thread_store_hides_assistant_claimed_by_foreign_binding():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_target")
    visible_message = store.append_message(
        "thread_target",
        role="user",
        content="Visible target message.",
    )
    claimed_message = store.append_message(
        "thread_target",
        role="assistant",
        content="FORGED TARGET ASSISTANT",
    )
    assert store.upsert_message_embedding(
        visible_message,
        embedding_model="visible-target-memory",
        vector=[1.0, 0.0],
    )
    assert store.upsert_message_embedding(
        claimed_message,
        embedding_model="foreign-claimed-memory",
        vector=[1.0, 0.0],
    )

    foreign_report = sample_loop_report(thread_id="thread_foreign")
    store.create_thread(thread_id="thread_foreign")
    store.append_turn(
        "thread_foreign",
        user_content=foreign_report.run.user_input,
        assistant_content=foreign_report.run.final_answer,
        raw_loop_report=foreign_report.to_dict(),
        public_loop_report=foreign_report.to_public_dict(),
    )
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_run_message_bindings
            SET assistant_message_id = ?
            WHERE run_id = ?
            """,
            (claimed_message.id, foreign_report.run.run_id),
        )
        store._conn.commit()

    detailed = store.get_thread("thread_target")
    recent = store.recent_messages("thread_target")
    listed = next(
        thread for thread in store.list_threads() if thread.id == "thread_target"
    )

    assert detailed is not None
    assert [(message.id, message.content) for message in detailed.messages] == [
        (visible_message.id, "Visible target message.")
    ]
    assert [(message.id, message.content) for message in recent] == [
        (visible_message.id, "Visible target message.")
    ]
    assert store.semantic_memories(
        "thread_target",
        embedding_model="foreign-claimed-memory",
        query_vector=[1.0, 0.0],
    ) == ()
    assert detailed.message_count == listed.message_count == 1
    assert detailed.memory_count == listed.memory_count == 1
    assert not store.has_message_embeddings(
        "thread_target",
        "foreign-claimed-memory",
    )
    assert store.has_message_embeddings(
        "thread_target",
        "visible-target-memory",
    )


@pytest.mark.parametrize("reused_field", ("user_message_id", "assistant_message_id"))
def test_thread_store_quarantines_all_pairs_when_binding_reuses_message_id(
    tmp_path,
    reused_field,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_ambiguous_binding")
    reports = (
        sample_loop_report(
            run_id="run_binding_first",
            thread_id="thread_ambiguous_binding",
        ),
        sample_loop_report(
            run_id="run_binding_second",
            thread_id="thread_ambiguous_binding",
        ),
    )
    persisted_turns = []
    for report in reports:
        persisted_turn = store.append_turn(
            "thread_ambiguous_binding",
            user_content=report.run.user_input,
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )
        assert persisted_turn is not None
        persisted_turns.append(persisted_turn)
        for message in persisted_turn:
            assert store.upsert_message_embedding(
                message,
                embedding_model="ambiguous-binding-memory",
                vector=[1.0, 0.0],
            )

    first_user, first_assistant = persisted_turns[0]
    reused_message_id = (
        first_user.id
        if reused_field == "user_message_id"
        else first_assistant.id
    )
    with store._lock:
        if reused_field == "user_message_id":
            store._conn.execute(
                """
                UPDATE loop_run_message_bindings
                SET user_message_id = ?
                WHERE run_id = ?
                """,
                (reused_message_id, reports[1].run.run_id),
            )
        else:
            store._conn.execute(
                """
                UPDATE loop_run_message_bindings
                SET assistant_message_id = ?
                WHERE run_id = ?
                """,
                (reused_message_id, reports[1].run.run_id),
            )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    records = restored.list_loop_runs("thread_ambiguous_binding")
    detailed = restored.get_thread("thread_ambiguous_binding")
    recent = restored.recent_messages("thread_ambiguous_binding")
    listed = next(
        thread
        for thread in restored.list_threads()
        if thread.id == "thread_ambiguous_binding"
    )

    assert {record.run_id for record in records} == {
        "run_binding_first",
        "run_binding_second",
    }
    assert all(
        record.quarantine_reason == "stored_message_binding_invalid"
        for record in records
    )
    for record in records:
        assert record.summary_dict()["projection_status"] == "quarantined"
        with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
            record.detail_dict()
    assert detailed is not None
    assert detailed.messages == ()
    assert recent == ()
    assert restored.semantic_memories(
        "thread_ambiguous_binding",
        embedding_model="ambiguous-binding-memory",
        query_vector=[1.0, 0.0],
    ) == ()
    assert detailed.message_count == listed.message_count == 0
    assert detailed.memory_count == listed.memory_count == 0
    assert detailed.loop_run_count == listed.loop_run_count == 2
    assert not restored.has_message_embeddings(
        "thread_ambiguous_binding",
        "ambiguous-binding-memory",
    )


def test_thread_store_quarantines_reused_message_pair_with_identical_content(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    first_report = sample_loop_report(
        run_id="run_first",
        thread_id="thread_local",
    )
    second_report = sample_loop_report(
        run_id="run_second",
        thread_id="thread_local",
    )
    for report in (first_report, second_report):
        store.append_turn(
            "thread_local",
            user_content=report.run.user_input,
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    first_binding = store._conn.execute(
        """
        SELECT user_message_id, assistant_message_id
        FROM loop_runs
        WHERE run_id = ?
        """,
        (first_report.run.run_id,),
    ).fetchone()
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_runs
            SET user_message_id = ?, assistant_message_id = ?
            WHERE run_id = ?
            """,
            (
                first_binding["user_message_id"],
                first_binding["assistant_message_id"],
                second_report.run.run_id,
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    records = restored.list_loop_runs("thread_local")

    assert {record.run_id for record in records} == {"run_first", "run_second"}
    assert all(
        record.quarantine_reason == "stored_message_binding_invalid"
        for record in records
    )
    assert all(
        record.summary_dict()["projection_status"] == "quarantined"
        for record in records
    )


def test_thread_store_quarantines_swapped_message_pairs_with_identical_content(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    reports = (
        sample_loop_report(run_id="run_first", thread_id="thread_local"),
        sample_loop_report(run_id="run_second", thread_id="thread_local"),
    )
    for report in reports:
        store.append_turn(
            "thread_local",
            user_content=report.run.user_input,
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    bindings = {
        str(row["run_id"]): (
            int(row["user_message_id"]),
            int(row["assistant_message_id"]),
        )
        for row in store._conn.execute(
            """
            SELECT run_id, user_message_id, assistant_message_id
            FROM loop_runs
            WHERE thread_id = ?
            """,
            ("thread_local",),
        ).fetchall()
    }
    first_user_id, first_assistant_id = bindings["run_first"]
    second_user_id, second_assistant_id = bindings["run_second"]
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_runs
            SET user_message_id = ?, assistant_message_id = ?
            WHERE run_id = ?
            """,
            (second_user_id, second_assistant_id, "run_first"),
        )
        store._conn.execute(
            """
            UPDATE loop_runs
            SET user_message_id = ?, assistant_message_id = ?
            WHERE run_id = ?
            """,
            (first_user_id, first_assistant_id, "run_second"),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    records = restored.list_loop_runs("thread_local")

    assert {record.run_id for record in records} == {"run_first", "run_second"}
    assert all(
        record.quarantine_reason == "stored_message_binding_invalid"
        for record in records
    )
    assert all(
        record.summary_dict()["projection_status"] == "quarantined"
        for record in records
    )


def test_thread_store_explicitly_quarantines_unbound_legacy_supported_row(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    report = legacy_unbound_supported_report()
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    with store._lock:
        store._conn.execute(
            """
            INSERT INTO loop_runs (
                run_id, thread_id, schema_version, raw_report_json,
                public_report_json, step_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.run.run_id,
                report.run.session_id,
                report.schema_version,
                json.dumps(report.to_dict()),
                json.dumps({"run": {"final_answer": "STALE_PUBLIC_ANSWER"}}),
                len(report.run.steps),
                "2026-01-01T00:00:00.030Z",
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    record = restored.get_loop_run("thread_local", "run_legacy_supported")

    assert record is not None
    assert record.quarantine_reason == "legacy_visible_answer_unbound"
    assert record.summary_dict()["quarantine_reason"] == (
        "legacy_visible_answer_unbound"
    )
    assert "STALE_PUBLIC_ANSWER" not in json.dumps(record.summary_dict())
    with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
        record.detail_dict()


def test_thread_store_explicitly_quarantines_unbound_legacy_not_verified_row(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    bound_report = sample_loop_report(
        run_id="run_legacy_not_verified",
        thread_id="thread_local",
    )
    legacy_final = replace(
        bound_report.run.steps[-1],
        metadata={},
    )
    report = replace(
        bound_report,
        run=replace(
            bound_report.run,
            steps=(*bound_report.run.steps[:-1], legacy_final),
        ),
    )
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    with store._lock:
        store._conn.execute(
            """
            INSERT INTO loop_runs (
                run_id, thread_id, schema_version, raw_report_json,
                public_report_json, step_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.run.run_id,
                report.run.session_id,
                report.schema_version,
                json.dumps(report.to_dict()),
                json.dumps({"run": {"final_answer": "STALE_PUBLIC_ANSWER"}}),
                len(report.run.steps),
                "2026-01-01T00:00:00.030Z",
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    record = restored.get_loop_run(
        "thread_local",
        "run_legacy_not_verified",
    )

    assert record is not None
    assert record.quarantine_reason == "legacy_visible_answer_unbound"
    assert "STALE_PUBLIC_ANSWER" not in json.dumps(record.summary_dict())
    with pytest.raises(QuarantinedLoopRunError, match="quarantined"):
        record.detail_dict()


def test_thread_store_replaces_hostile_timestamps_with_one_bounded_fallback(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    quarantined_report = sample_loop_report(
        run_id="run_hostile_timestamp",
        thread_id="thread_quarantined_timestamp",
    )
    store.create_thread(thread_id="thread_quarantined_timestamp")
    store.append_turn(
        "thread_quarantined_timestamp",
        user_content=quarantined_report.run.user_input,
        assistant_content=quarantined_report.run.final_answer,
        raw_loop_report=quarantined_report.to_dict(),
        public_loop_report=quarantined_report.to_public_dict(),
    )

    valid_report = sample_loop_report(
        run_id="run_valid_timestamp",
        thread_id="thread_valid_timestamp",
    )
    store.create_thread(thread_id="thread_valid_timestamp")
    valid_turn = store.append_turn(
        "thread_valid_timestamp",
        user_content=valid_report.run.user_input,
        assistant_content=valid_report.run.final_answer,
        raw_loop_report=valid_report.to_dict(),
        public_loop_report=valid_report.to_public_dict(),
    )
    assert valid_turn is not None
    for message in valid_turn:
        assert store.upsert_message_embedding(
            message,
            embedding_model="hostile-timestamp-memory",
            vector=[1.0, 0.0],
        )

    control_timestamp = (
        "2026-01-01T00:00:00.000Z\x00CONTROL_TIMESTAMP_SECRET"
    )
    unbounded_timestamp = "9" * 4096 + "UNBOUNDED_TIMESTAMP_SECRET"
    malformed_timestamp = "MALFORMED_TIMESTAMP_SECRET"
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_runs
            SET raw_report_json = ?, created_at = ?
            WHERE run_id = ?
            """,
            (
                '{"forged":true}',
                control_timestamp,
                quarantined_report.run.run_id,
            ),
        )
        store._conn.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            (unbounded_timestamp, valid_turn[0].id),
        )
        store._conn.execute(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            (malformed_timestamp, valid_turn[1].id),
        )
        store._conn.execute(
            """
            UPDATE threads
            SET created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                control_timestamp,
                unbounded_timestamp,
                "thread_valid_timestamp",
            ),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    direct_run = restored.get_loop_run(
        "thread_quarantined_timestamp",
        quarantined_report.run.run_id,
    )
    listed_run = restored.list_loop_runs("thread_quarantined_timestamp")[0]
    quarantined_thread = restored.get_thread("thread_quarantined_timestamp")
    detailed = restored.get_thread("thread_valid_timestamp")
    recent = restored.recent_messages("thread_valid_timestamp")
    memories = restored.semantic_memories(
        "thread_valid_timestamp",
        embedding_model="hostile-timestamp-memory",
        query_vector=[1.0, 0.0],
    )
    listed_thread = next(
        thread
        for thread in restored.list_threads()
        if thread.id == "thread_valid_timestamp"
    )

    assert direct_run is not None
    assert direct_run.quarantine_reason == "stored_loop_report_invalid"
    assert quarantined_thread is not None
    assert detailed is not None
    assert len(detailed.messages) == len(recent) == len(memories) == 2
    fallback_values = (
        direct_run.created_at,
        direct_run.summary_dict()["created_at"],
        listed_run.created_at,
        quarantined_thread.loop_runs[0].created_at,
        *(message.created_at for message in detailed.messages),
        *(message.created_at for message in recent),
        *(memory.created_at for memory in memories),
        detailed.created_at,
        detailed.updated_at,
        listed_thread.created_at,
        listed_thread.updated_at,
    )
    assert len(set(fallback_values)) == 1
    fallback = fallback_values[0]
    assert len(fallback.encode("utf-8")) <= 32
    parsed_fallback = datetime.fromisoformat(
        fallback.replace("Z", "+00:00")
    )
    assert fallback == (
        parsed_fallback.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    exposed_payload = json.dumps(
        {
            "run": direct_run.summary_dict(),
            "thread": detailed.detail_dict(),
            "recent": [message.to_dict() for message in recent],
            "memories": [memory.created_at for memory in memories],
            "listed": listed_thread.summary_dict(),
        }
    )
    for hostile_marker in (
        "CONTROL_TIMESTAMP_SECRET",
        "UNBOUNDED_TIMESTAMP_SECRET",
        "MALFORMED_TIMESTAMP_SECRET",
    ):
        assert hostile_marker not in exposed_payload


def test_thread_store_orders_tied_loop_runs_by_latest_insertion_after_restart(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    store.create_thread(thread_id="thread_local")
    for run_id in ("run_first", "run_second"):
        report = sample_loop_report(run_id=run_id, thread_id="thread_local")
        store.append_turn(
            "thread_local",
            user_content=report.run.user_input,
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )
    with store._lock:
        store._conn.execute(
            "UPDATE loop_runs SET created_at = ? WHERE thread_id = ?",
            ("2026-01-01T00:00:00.000Z", "thread_local"),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)

    assert [run.run_id for run in restored.list_loop_runs("thread_local")] == [
        "run_second",
        "run_first",
    ]
    assert [
        run.run_id for run in restored.get_thread("thread_local").loop_runs
    ] == ["run_second", "run_first"]


def test_thread_store_orders_latest_loop_run_by_insertion_when_clock_rolls_back(
    monkeypatch,
):
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    clock = iter(
        (
            "2026-01-02T00:00:00.000Z",
            "2026-01-01T00:00:00.000Z",
        )
    )
    monkeypatch.setattr("src.thread_store.utc_now", lambda: next(clock))
    for run_id in ("run_first", "run_second"):
        report = sample_loop_report(run_id=run_id, thread_id="thread_local")
        store.append_turn(
            "thread_local",
            user_content=report.run.user_input,
            assistant_content=report.run.final_answer,
            raw_loop_report=report.to_dict(),
            public_loop_report=report.to_public_dict(),
        )

    assert [run.run_id for run in store.list_loop_runs("thread_local")] == [
        "run_second",
        "run_first",
    ]
    assert [
        run.run_id for run in store.get_thread("thread_local").loop_runs
    ] == ["run_second", "run_first"]


def test_thread_store_clear_removes_loop_runs():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")
    store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
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


def test_thread_store_atomically_reports_single_thread_creator():
    store = ThreadStore.in_memory()
    start = threading.Barrier(2)
    results = []

    def acquire():
        start.wait(timeout=5)
        results.append(store.ensure_thread_with_created("thread_race"))

    workers = [threading.Thread(target=acquire) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert sorted(created for _thread, created in results) == [False, True]
    assert len({thread.instance_id for thread, _created in results}) == 1


def test_thread_store_creation_ownership_is_atomic_across_connections(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    stores = [ThreadStore(db_path), ThreadStore(db_path)]
    start = threading.Barrier(2)
    results = []

    def acquire(store):
        start.wait(timeout=5)
        results.append(store.ensure_thread_with_created("thread_race"))

    workers = [
        threading.Thread(target=acquire, args=(store,))
        for store in stores
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert sorted(created for _thread, created in results) == [False, True]
    assert len({thread.instance_id for thread, _created in results}) == 1
    for store in stores:
        store.close()


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


@pytest.mark.parametrize(
    "expected_instance_id, expected_generation",
    (("instance_only", None), (None, 0)),
)
def test_thread_store_clear_rejects_partial_expected_identity(
    expected_instance_id,
    expected_generation,
):
    store = ThreadStore.in_memory()

    with pytest.raises(ValueError, match="must be supplied together"):
        store.clear_thread(
            "thread_local",
            expected_instance_id=expected_instance_id,
            expected_generation=expected_generation,
        )


@pytest.mark.parametrize("stale_field", ("instance_id", "generation"))
def test_thread_store_clear_preserves_thread_when_expected_identity_is_stale(
    stale_field,
):
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_clear_expected")
    store.append_message(
        thread.id,
        role="user",
        content="Replacement state must survive a stale clear.",
    )
    expected_instance_id = thread.instance_id
    expected_generation = thread.generation
    if stale_field == "instance_id":
        expected_instance_id = "stale_instance"
    else:
        expected_generation += 1

    cleared = store.clear_thread(
        thread.id,
        expected_instance_id=expected_instance_id,
        expected_generation=expected_generation,
    )

    assert cleared is None
    current = store.get_thread(thread.id)
    assert current is not None
    assert current.instance_id == thread.instance_id
    assert current.generation == thread.generation
    assert [message.content for message in current.messages] == [
        "Replacement state must survive a stale clear."
    ]


@pytest.mark.parametrize(
    "expected_instance_id, expected_generation",
    (("instance_only", None), (None, 0)),
)
def test_thread_store_delete_rejects_partial_expected_identity(
    expected_instance_id,
    expected_generation,
):
    store = ThreadStore.in_memory()

    with pytest.raises(ValueError, match="must be supplied together"):
        store.delete_thread(
            "thread_local",
            expected_instance_id=expected_instance_id,
            expected_generation=expected_generation,
        )


@pytest.mark.parametrize("stale_field", ("instance_id", "generation"))
def test_thread_store_delete_preserves_thread_when_expected_identity_is_stale(
    stale_field,
):
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_delete_expected")
    store.append_message(
        thread.id,
        role="user",
        content="Replacement state must survive a stale delete.",
    )
    expected_instance_id = thread.instance_id
    expected_generation = thread.generation
    if stale_field == "instance_id":
        expected_instance_id = "stale_instance"
    else:
        expected_generation += 1

    deleted = store.delete_thread(
        thread.id,
        expected_instance_id=expected_instance_id,
        expected_generation=expected_generation,
    )

    assert deleted is False
    current = store.get_thread(thread.id)
    assert current is not None
    assert current.instance_id == thread.instance_id
    assert current.generation == thread.generation
    assert [message.content for message in current.messages] == [
        "Replacement state must survive a stale delete."
    ]


def test_thread_store_delete_expected_identity_does_not_target_aba_replacement():
    store = ThreadStore.in_memory()
    old_thread = store.create_thread(thread_id="thread_delete_aba")
    assert store.delete_thread(old_thread.id) is True
    replacement = store.create_thread(thread_id=old_thread.id)
    store.append_message(
        replacement.id,
        role="user",
        content="ABA replacement must survive stale delete.",
    )

    deleted = store.delete_thread(
        old_thread.id,
        expected_instance_id=old_thread.instance_id,
        expected_generation=old_thread.generation,
    )

    assert deleted is False
    current = store.get_thread(old_thread.id)
    assert current is not None
    assert current.instance_id == replacement.instance_id
    assert current.instance_id != old_thread.instance_id
    assert [message.content for message in current.messages] == [
        "ABA replacement must survive stale delete."
    ]


def test_thread_store_deletes_only_matching_current_empty_thread():
    store = ThreadStore.in_memory()
    empty_thread = store.create_thread(thread_id="thread_empty")

    assert (
        store.delete_empty_thread_if_current(
            "thread_empty",
            expected_instance_id="wrong_instance",
            expected_generation=empty_thread.generation,
            expected_updated_at=empty_thread.updated_at,
            expected_title=empty_thread.title,
        )
        is False
    )
    assert (
        store.delete_empty_thread_if_current(
            "thread_empty",
            expected_instance_id=empty_thread.instance_id,
            expected_generation=empty_thread.generation + 1,
            expected_updated_at=empty_thread.updated_at,
            expected_title=empty_thread.title,
        )
        is False
    )
    assert store.get_thread("thread_empty") is not None
    assert (
        store.delete_empty_thread_if_current(
            "thread_empty",
            expected_instance_id=empty_thread.instance_id,
            expected_generation=empty_thread.generation,
            expected_updated_at=empty_thread.updated_at,
            expected_title=empty_thread.title,
        )
        is True
    )
    assert store.get_thread("thread_empty") is None

    nonempty_thread = store.create_thread(thread_id="thread_nonempty")
    store.append_message("thread_nonempty", role="user", content="keep me")
    assert (
        store.delete_empty_thread_if_current(
            "thread_nonempty",
            expected_instance_id=nonempty_thread.instance_id,
            expected_generation=nonempty_thread.generation,
            expected_updated_at=nonempty_thread.updated_at,
            expected_title=nonempty_thread.title,
        )
        is False
    )
    assert store.get_thread("thread_nonempty").message_count == 1

    renamed_thread = store.create_thread(thread_id="thread_renamed")
    store.rename_thread("thread_renamed", "User-owned name")
    assert (
        store.delete_empty_thread_if_current(
            "thread_renamed",
            expected_instance_id=renamed_thread.instance_id,
            expected_generation=renamed_thread.generation,
            expected_updated_at=renamed_thread.updated_at,
            expected_title=renamed_thread.title,
        )
        is False
    )
    assert store.get_thread("thread_renamed").title == "User-owned name"


def test_thread_store_delete_removes_messages_runs_and_memories():
    store = ThreadStore.in_memory()
    report = sample_loop_report(thread_id="thread_local")
    store.create_thread(thread_id="thread_local")
    messages = store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
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


def test_thread_store_append_turn_serializes_cross_connection_clear(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    writer = ThreadStore(db_path)
    competing = ThreadStore(db_path)
    thread = writer.create_thread(thread_id="thread_race_clear")
    competing._conn.execute("PRAGMA busy_timeout = 1")

    def clear_during_guard_check():
        try:
            competing.clear_thread(thread.id)
        except sqlite3.OperationalError as exc:
            competing._conn.rollback()
            assert "locked" in str(exc).lower()
            return False
        return True

    proxy = _GuardCheckInterleavingConnection(
        writer._conn,
        clear_during_guard_check,
    )
    writer._conn = proxy

    persisted_turn = writer.append_turn(
        thread.id,
        user_content="Current question",
        assistant_content="Current answer",
        expected_generation=thread.generation,
        expected_instance_id=thread.instance_id,
    )

    assert proxy.mutation_committed is False
    assert persisted_turn is not None
    cleared = competing.clear_thread(thread.id)
    assert cleared is not None
    assert cleared.generation == thread.generation + 1
    assert cleared.messages == ()


def test_thread_store_clear_serializes_cross_connection_aba_recreate(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    clearer = ThreadStore(db_path)
    competing = ThreadStore(db_path)
    old_thread = clearer.create_thread(thread_id="thread_clear_aba")
    clearer.append_message(
        old_thread.id,
        role="user",
        content="Old instance message",
    )
    competing._conn.execute("PRAGMA busy_timeout = 1")

    def recreate_during_identity_check():
        try:
            assert competing.delete_thread(old_thread.id) is True
            replacement = competing.create_thread(thread_id=old_thread.id)
            competing.append_message(
                replacement.id,
                role="user",
                content="Replacement message must survive",
            )
        except sqlite3.OperationalError as exc:
            competing._conn.rollback()
            assert "locked" in str(exc).lower()
            return False
        return True

    proxy = _GuardCheckInterleavingConnection(
        clearer._conn,
        recreate_during_identity_check,
        sql_markers=("SELECT instance_id, generation", "FROM threads", "WHERE id = ?"),
    )
    clearer._conn = proxy

    cleared = clearer.clear_thread(old_thread.id)

    assert proxy.mutation_committed is True
    assert cleared is None
    current = clearer.get_thread(old_thread.id)
    assert current is not None
    assert current.instance_id != old_thread.instance_id
    assert [message.content for message in current.messages] == [
        "Replacement message must survive"
    ]


def test_thread_store_append_turn_serializes_cross_connection_aba_recreate(
    tmp_path,
):
    db_path = tmp_path / "threads.sqlite3"
    writer = ThreadStore(db_path)
    competing = ThreadStore(db_path)
    old_thread = writer.create_thread(thread_id="thread_race_aba")
    competing._conn.execute("PRAGMA busy_timeout = 1")

    def recreate_during_guard_check():
        try:
            assert competing.delete_thread(old_thread.id) is True
            competing.create_thread(thread_id=old_thread.id)
        except sqlite3.OperationalError as exc:
            competing._conn.rollback()
            assert "locked" in str(exc).lower()
            return False
        return True

    proxy = _GuardCheckInterleavingConnection(
        writer._conn,
        recreate_during_guard_check,
    )
    writer._conn = proxy

    persisted_turn = writer.append_turn(
        old_thread.id,
        user_content="Current question",
        assistant_content="Current answer",
        expected_generation=old_thread.generation,
        expected_instance_id=old_thread.instance_id,
    )

    assert proxy.mutation_committed is False
    assert persisted_turn is not None
    assert competing.delete_thread(old_thread.id) is True
    replacement = competing.create_thread(thread_id=old_thread.id)
    assert replacement.instance_id != old_thread.instance_id
    assert replacement.messages == ()


def test_thread_store_append_turn_skips_loop_run_when_generation_changed():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_local")
    report = sample_loop_report(thread_id="thread_local")

    store.clear_thread("thread_local")
    result = store.append_turn(
        "thread_local",
        user_content=report.run.user_input,
        assistant_content=report.run.final_answer,
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


@pytest.mark.parametrize(
    "poisoned_metadata_json",
    (
        '{"safe":true,"safe":false}',
        '{"score":NaN}',
        '["not-an-object"]',
    ),
)
def test_thread_store_suppresses_malformed_optional_recipe_metadata_on_restart(
    tmp_path,
    poisoned_metadata_json,
):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    recipe = store.create_recipe(
        recipe_id="recipe_corrupt_metadata",
        name="Durable recipe",
        goal="Stay readable after optional metadata corruption.",
        metadata={"valid": True},
    )
    with store._lock:
        store._conn.execute(
            "UPDATE loop_recipes SET metadata_json = ? WHERE recipe_id = ?",
            (poisoned_metadata_json, recipe.recipe_id),
        )
        store._conn.commit()
    store.close()

    restored = ThreadStore(db_path)
    restored_recipe = restored.get_recipe(recipe.recipe_id)

    assert restored_recipe is not None
    assert restored_recipe.name == "Durable recipe"
    assert restored_recipe.goal == "Stay readable after optional metadata corruption."
    assert restored_recipe.metadata == {}
    assert recipe.recipe_id in {
        listed_recipe.recipe_id for listed_recipe in restored.list_recipes()
    }


@pytest.mark.parametrize(
    "poisoned_success_criteria_json",
    (
        "[NaN]",
        '["valid",{"not":"a string"}]',
        "not-json",
    ),
)
def test_thread_store_fails_safe_for_malformed_recipe_success_criteria(
    poisoned_success_criteria_json,
):
    store = ThreadStore.in_memory()
    store.ensure_default_recipe()
    recipe = store.create_recipe(
        recipe_id="recipe_corrupt_criteria",
        name="Durable recipe",
        goal="Stay readable after criteria corruption.",
        success_criteria=("Initially valid",),
    )
    with store._lock:
        store._conn.execute(
            """
            UPDATE loop_recipes
            SET success_criteria_json = ?
            WHERE recipe_id IN (?, ?)
            """,
            (
                poisoned_success_criteria_json,
                recipe.recipe_id,
                DEFAULT_LOOP_RECIPE_ID,
            ),
        )
        store._conn.commit()

    restored_recipe = store.get_recipe(recipe.recipe_id)
    listed = store.list_recipes()
    default_recipe = store.ensure_default_recipe()

    assert restored_recipe is not None
    assert restored_recipe.success_criteria == ()
    assert recipe.recipe_id in {listed_recipe.recipe_id for listed_recipe in listed}
    assert default_recipe.recipe_id == DEFAULT_LOOP_RECIPE_ID
    assert default_recipe.success_criteria


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
