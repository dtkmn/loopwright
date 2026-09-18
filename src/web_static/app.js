const ACTIVE_THREAD_STORAGE_KEY = "loopwright.active-thread.v1";
const ACTIVE_RECIPE_STORAGE_KEY = "loopwright.active-recipe.v1";
const LEGACY_ACTIVE_THREAD_STORAGE_KEY = "ai-loop-engine.active-thread.v1";
const LEGACY_ACTIVE_RECIPE_STORAGE_KEY = "ai-loop-engine.active-recipe.v1";
const LEGACY_THREAD_STORAGE_KEY = "ai-loop-engine.threads.v1";
const DEFAULT_THREAD_TITLE = "New thread";
const MAX_THREADS = 30;
const MAX_THREAD_MESSAGES = 100;
const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$/;
const QUERY_PROGRESS_INTERVAL_MS = 1000;
const DEFAULT_TEXT_ENCODING = "auto";
const PUBLIC_REPORT_SOURCE_SCHEMA = "loop-report/v1";
const PUBLIC_REPORT_PROJECTION_SCHEMA = "loop-public-report/v1";
const PUBLIC_REDACTION_REASON = "terminal_public_redaction";
const PUBLIC_REDACTION_TEXT = "[redacted: terminal decision]";
const SELF_CHECK_REFUSAL_ANSWER =
  "I could not find enough relevant information in the provided evidence to answer that.";
const PUBLIC_EVIDENCE_ID_PATTERN = /^evidence_[0-9a-f]{64}$/;
const PUBLIC_CATEGORY_PATTERN = /^[a-z][a-z0-9_.:-]{0,63}$/;
const PUBLIC_CONTEXT_PROVIDERS = new Set(["document", "none", "web"]);
const PUBLIC_EVIDENCE_PROVIDERS = new Set(["document", "web"]);
const PUBLIC_BACKENDS = new Set([
  "auto",
  "mock",
  "ollama",
  "openai-compatible",
]);
const PUBLIC_MEMORY_STATUSES = new Set([
  "empty",
  "not_requested",
  "retrieved",
  "unavailable",
]);
// Keep display-label blankness identical to the Python public projector.
// Language-native trim functions disagree on NEL (U+0085) and BOM (U+FEFF).
const PUBLIC_DISPLAY_BLANK_CODE_POINTS = new Set([
  0x0020,
  0x0085,
  0x00a0,
  0x1680,
  0x2000,
  0x2001,
  0x2002,
  0x2003,
  0x2004,
  0x2005,
  0x2006,
  0x2007,
  0x2008,
  0x2009,
  0x200a,
  0x2028,
  0x2029,
  0x202f,
  0x205f,
  0x3000,
  0xfeff,
]);
// Unicode 15.0 General_Category=Cf ranges. Keep this explicit and mirrored in
// the Python projector so runtime Unicode-table versions cannot diverge.
const PUBLIC_DISPLAY_FORMAT_CODE_POINT_RANGES = Object.freeze([
  [0x00ad, 0x00ad],
  [0x0600, 0x0605],
  [0x061c, 0x061c],
  [0x06dd, 0x06dd],
  [0x070f, 0x070f],
  [0x0890, 0x0891],
  [0x08e2, 0x08e2],
  [0x180e, 0x180e],
  [0x200b, 0x200f],
  [0x202a, 0x202e],
  [0x2060, 0x2064],
  [0x2066, 0x206f],
  [0xfeff, 0xfeff],
  [0xfff9, 0xfffb],
  [0x110bd, 0x110bd],
  [0x110cd, 0x110cd],
  [0x13430, 0x1343f],
  [0x1bca0, 0x1bca3],
  [0x1d173, 0x1d17a],
  [0xe0001, 0xe0001],
  [0xe0020, 0xe007f],
]);
// Unicode 15.0.0 General_Category=M (Mn, Mc, or Me). This exact token list
// mirrors the Python projector. Marks are ignorable only for deciding whether
// a label has a visible base; visible-base-plus-mark labels remain valid.
const PUBLIC_DISPLAY_MARK_UNICODE_VERSION = "15.0.0";
const PUBLIC_DISPLAY_MARK_CODE_POINT_RANGES = Object.freeze(
  `
0300-036F 0483-0489 0591-05BD 05BF 05C1-05C2 05C4-05C5 05C7 0610-061A 064B-065F 0670
06D6-06DC 06DF-06E4 06E7-06E8 06EA-06ED 0711 0730-074A 07A6-07B0 07EB-07F3 07FD 0816-0819
081B-0823 0825-0827 0829-082D 0859-085B 0898-089F 08CA-08E1 08E3-0903 093A-093C 093E-094F 0951-0957
0962-0963 0981-0983 09BC 09BE-09C4 09C7-09C8 09CB-09CD 09D7 09E2-09E3 09FE 0A01-0A03
0A3C 0A3E-0A42 0A47-0A48 0A4B-0A4D 0A51 0A70-0A71 0A75 0A81-0A83 0ABC 0ABE-0AC5
0AC7-0AC9 0ACB-0ACD 0AE2-0AE3 0AFA-0AFF 0B01-0B03 0B3C 0B3E-0B44 0B47-0B48 0B4B-0B4D 0B55-0B57
0B62-0B63 0B82 0BBE-0BC2 0BC6-0BC8 0BCA-0BCD 0BD7 0C00-0C04 0C3C 0C3E-0C44 0C46-0C48
0C4A-0C4D 0C55-0C56 0C62-0C63 0C81-0C83 0CBC 0CBE-0CC4 0CC6-0CC8 0CCA-0CCD 0CD5-0CD6 0CE2-0CE3
0CF3 0D00-0D03 0D3B-0D3C 0D3E-0D44 0D46-0D48 0D4A-0D4D 0D57 0D62-0D63 0D81-0D83 0DCA
0DCF-0DD4 0DD6 0DD8-0DDF 0DF2-0DF3 0E31 0E34-0E3A 0E47-0E4E 0EB1 0EB4-0EBC 0EC8-0ECE
0F18-0F19 0F35 0F37 0F39 0F3E-0F3F 0F71-0F84 0F86-0F87 0F8D-0F97 0F99-0FBC 0FC6
102B-103E 1056-1059 105E-1060 1062-1064 1067-106D 1071-1074 1082-108D 108F 109A-109D 135D-135F
1712-1715 1732-1734 1752-1753 1772-1773 17B4-17D3 17DD 180B-180D 180F 1885-1886 18A9
1920-192B 1930-193B 1A17-1A1B 1A55-1A5E 1A60-1A7C 1A7F 1AB0-1ACE 1B00-1B04 1B34-1B44 1B6B-1B73
1B80-1B82 1BA1-1BAD 1BE6-1BF3 1C24-1C37 1CD0-1CD2 1CD4-1CE8 1CED 1CF4 1CF7-1CF9 1DC0-1DFF
20D0-20F0 2CEF-2CF1 2D7F 2DE0-2DFF 302A-302F 3099-309A A66F-A672 A674-A67D A69E-A69F A6F0-A6F1
A802 A806 A80B A823-A827 A82C A880-A881 A8B4-A8C5 A8E0-A8F1 A8FF A926-A92D
A947-A953 A980-A983 A9B3-A9C0 A9E5 AA29-AA36 AA43 AA4C-AA4D AA7B-AA7D AAB0 AAB2-AAB4
AAB7-AAB8 AABE-AABF AAC1 AAEB-AAEF AAF5-AAF6 ABE3-ABEA ABEC-ABED FB1E FE00-FE0F FE20-FE2F
101FD 102E0 10376-1037A 10A01-10A03 10A05-10A06 10A0C-10A0F 10A38-10A3A 10A3F 10AE5-10AE6 10D24-10D27
10EAB-10EAC 10EFD-10EFF 10F46-10F50 10F82-10F85 11000-11002 11038-11046 11070 11073-11074 1107F-11082 110B0-110BA
110C2 11100-11102 11127-11134 11145-11146 11173 11180-11182 111B3-111C0 111C9-111CC 111CE-111CF 1122C-11237
1123E 11241 112DF-112EA 11300-11303 1133B-1133C 1133E-11344 11347-11348 1134B-1134D 11357 11362-11363
11366-1136C 11370-11374 11435-11446 1145E 114B0-114C3 115AF-115B5 115B8-115C0 115DC-115DD 11630-11640 116AB-116B7
1171D-1172B 1182C-1183A 11930-11935 11937-11938 1193B-1193E 11940 11942-11943 119D1-119D7 119DA-119E0 119E4
11A01-11A0A 11A33-11A39 11A3B-11A3E 11A47 11A51-11A5B 11A8A-11A99 11C2F-11C36 11C38-11C3F 11C92-11CA7 11CA9-11CB6
11D31-11D36 11D3A 11D3C-11D3D 11D3F-11D45 11D47 11D8A-11D8E 11D90-11D91 11D93-11D97 11EF3-11EF6 11F00-11F01
11F03 11F34-11F3A 11F3E-11F42 13440 13447-13455 16AF0-16AF4 16B30-16B36 16F4F 16F51-16F87 16F8F-16F92
16FE4 16FF0-16FF1 1BC9D-1BC9E 1CF00-1CF2D 1CF30-1CF46 1D165-1D169 1D16D-1D172 1D17B-1D182 1D185-1D18B 1D1AA-1D1AD
1D242-1D244 1DA00-1DA36 1DA3B-1DA6C 1DA75 1DA84 1DA9B-1DA9F 1DAA1-1DAAF 1E000-1E006 1E008-1E018 1E01B-1E021
1E023-1E024 1E026-1E02A 1E08F 1E130-1E136 1E2AE 1E2EC-1E2EF 1E4EC-1E4EF 1E8D0-1E8D6 1E944-1E94A E0100-E01EF
  `.trim().split(/\s+/).map((token) => {
    const [start, end = start] = token.split("-");
    return [Number.parseInt(start, 16), Number.parseInt(end, 16)];
  }),
);
const LOOP_RUN_PROJECTION_STATUSES = new Set(["available", "quarantined"]);
const DEFAULT_QUARANTINE_REASON = "stored_loop_report_invalid";
const LOOP_RUN_QUARANTINE_REASON_LABELS = new Map([
  ["stored_loop_report_invalid", "Stored report failed validation."],
  [
    "legacy_visible_answer_unbound",
    "Legacy visible answer lacks safe answer binding.",
  ],
]);
const AVAILABLE_LOOP_RUN_SUMMARY_KEYS = Object.freeze([
  "run_id",
  "thread_id",
  "projection_status",
  "projection_schema_version",
  "final_decision",
  "terminal_reason",
  "context_provider",
  "backend",
  "model",
  "step_count",
  "started_at",
  "completed_at",
  "created_at",
]);
const QUARANTINED_LOOP_RUN_SUMMARY_KEYS = Object.freeze([
  "run_id",
  "thread_id",
  "created_at",
  "projection_status",
  "quarantine_reason",
]);
const PUBLIC_DETAIL_KEYS = Object.freeze([
  "run_id",
  "thread_id",
  "projection_status",
  "projection_schema_version",
  "final_decision",
  "terminal_reason",
  "context_provider",
  "backend",
  "model",
  "step_count",
  "started_at",
  "completed_at",
  "created_at",
  "report",
  "public",
]);
const PUBLIC_REPORT_KEYS = Object.freeze([
  "schema_version",
  "projection_schema_version",
  "public",
  "public_redaction",
  "run",
]);
const PUBLIC_REDACTION_KEYS = Object.freeze(["applied", "reason"]);
const PUBLIC_RUN_KEYS = Object.freeze([
  "run_id",
  "session_id",
  "context_provider",
  "conversation_context_count",
  "semantic_memory_count",
  "semantic_memory_status",
  "backend",
  "model_label",
  "policy",
  "started_at",
  "completed_at",
  "steps",
  "evidence",
  "final_decision",
  "terminal_reason",
  "final_answer",
  "error_present",
]);
const PUBLIC_POLICY_KEYS = Object.freeze([
  "max_retries",
  "require_citations",
  "require_verifier_for_supported",
  "allow_mock_supported",
  "allow_tool_calls",
  "require_human_review_for_tools",
]);
const PUBLIC_STEP_KEYS = Object.freeze([
  "step_id",
  "phase",
  "decision",
  "started_at",
  "ended_at",
  "duration_ms",
  "backend",
  "model_label",
  "retry_count",
  "error_present",
  "verification",
  "human_review_required",
]);
const PUBLIC_VERIFICATION_KEYS = Object.freeze([
  "outcome",
  "verifier_backend",
  "verifier_model_label",
  "same_model_as_drafter",
]);
const PUBLIC_EVIDENCE_KEYS = Object.freeze([
  "evidence_id",
  "citation_id",
  "provider",
  "locator",
]);
const PUBLIC_EVIDENCE_LOCATOR_KEYS = Object.freeze(["page", "chunk_index"]);
const PUBLIC_PHASES = new Set([
  "input",
  "context_select",
  "retrieve",
  "draft",
  "format_check",
  "mechanical_check",
  "verify",
  "retry",
  "refuse",
  "final",
  "error",
]);
const PUBLIC_DECISIONS = new Set([
  "continue",
  "retry",
  "refuse",
  "block",
  "requires_review",
  "supported",
  "not_verified",
  "final",
  "error",
]);
const PUBLIC_FINAL_DECISIONS = new Set([
  "final",
  "supported",
  "not_verified",
  "refuse",
  "block",
  "requires_review",
  "error",
]);
const PUBLIC_TERMINAL_DECISIONS = new Set([
  "refuse",
  "block",
  "requires_review",
]);
const PUBLIC_FINAL_REASON_CONTRACT = new Map([
  [null, new Set([null])],
  ["final", new Set(["completed", "unspecified"])],
  ["supported", new Set(["completed", "unspecified"])],
  ["not_verified", new Set(["not_verified", "trace_unavailable", "unspecified"])],
  [
    "refuse",
    new Set([
      "verification_failed",
      "retry_budget_exhausted",
      "policy_refused",
      "unspecified",
    ]),
  ],
  ["block", new Set(["blocked", "unspecified"])],
  ["requires_review", new Set(["human_review_required", "unspecified"])],
  [
    "error",
    new Set([
      "error",
      "retry_budget_exhausted",
      "trace_unavailable",
      "unspecified",
    ]),
  ],
]);
const PUBLIC_TERMINAL_REASONS = new Set([
  "completed",
  "not_verified",
  "verification_failed",
  "retry_budget_exhausted",
  "trace_unavailable",
  "policy_refused",
  "blocked",
  "human_review_required",
  "error",
  "unspecified",
]);
const PUBLIC_VERIFICATION_OUTCOMES = new Set([
  "supported",
  "unsupported",
  "insufficient",
  "not_verified",
  "error",
]);

const state = {
  threads: [],
  recipes: [],
  activeThreadId: null,
  activeRecipeId: null,
  recipeDraft: null,
  latest: null,
  runningQuery: null,
  threadSwitchRequestId: 0,
  threadOperationRequestId: 0,
  runInspection: {
    requestId: 0,
    threadId: null,
    runId: null,
    pendingRunId: null,
    status: "idle",
    message: "",
  },
  deletingThreadTokens: new Map(),
  clearingThreadTokens: new Map(),
  uploadThreadTokens: new Map(),
};

const elements = {
  backendPill: document.querySelector("#backend-pill"),
  modelPill: document.querySelector("#model-pill"),
  readyPill: document.querySelector("#ready-pill"),
  uploadButton: document.querySelector("#upload-button"),
  uploadStatus: document.querySelector("#upload-status"),
  fileInput: document.querySelector("#document-file"),
  textEncoding: document.querySelector("#text-encoding"),
  newThreadButton: document.querySelector("#new-thread"),
  deleteThreadButton: document.querySelector("#delete-thread"),
  threadList: document.querySelector("#thread-list"),
  recipeSelect: document.querySelector("#recipe-select"),
  recipeStatus: document.querySelector("#recipe-status"),
  recipeState: document.querySelector("#recipe-state"),
  recipeEditor: document.querySelector("#recipe-editor"),
  recipeEditorState: document.querySelector("#recipe-editor-state"),
  recipeNewButton: document.querySelector("#recipe-new"),
  recipeSaveButton: document.querySelector("#recipe-save"),
  recipeDeleteButton: document.querySelector("#recipe-delete"),
  recipeExportButton: document.querySelector("#recipe-export"),
  recipeImport: document.querySelector("#recipe-import"),
  recipeName: document.querySelector("#recipe-name"),
  recipeGoal: document.querySelector("#recipe-goal"),
  recipeInstructions: document.querySelector("#recipe-instructions"),
  recipeCriteria: document.querySelector("#recipe-criteria"),
  recipeStop: document.querySelector("#recipe-stop"),
  activeThreadTitle: document.querySelector("#active-thread-title"),
  activeThreadMemory: document.querySelector("#active-thread-memory"),
  fileScope: document.querySelector("#file-scope"),
  refreshStatus: document.querySelector("#refresh-status"),
  queryForm: document.querySelector("#query-form"),
  queryContext: document.querySelector("#query-context"),
  queryInput: document.querySelector("#query-input"),
  queryButton: document.querySelector("#query-button"),
  clearButton: document.querySelector("#clear-chat"),
  messages: document.querySelector("#messages"),
  memoryStatus: document.querySelector("#memory-status"),
  timeline: document.querySelector("#timeline"),
  runList: document.querySelector("#run-list"),
  runCount: document.querySelector("#run-count"),
  finalDecision: document.querySelector("#final-decision"),
  thinkingPanel: document.querySelector("#thinking-panel"),
  thinkingState: document.querySelector("#thinking-state"),
  thinkingNote: document.querySelector("#thinking-note"),
  thinkingContent: document.querySelector("#thinking-content"),
  summaryJson: document.querySelector("#summary-json"),
  traceJson: document.querySelector("#trace-json"),
  tabButtons: document.querySelectorAll(".tab-button"),
};

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) {
    button.textContent = busy ? "Working..." : label;
  }
}

function setQueryControlsBusy(busy) {
  elements.queryButton.disabled = busy;
  elements.queryButton.textContent = busy ? "Thinking..." : "Ask";
  elements.queryInput.disabled = busy;
  if (elements.queryContext) {
    elements.queryContext.disabled = busy;
  }
  elements.queryForm.dataset.running = String(Boolean(busy));
  elements.messages.dataset.running = String(Boolean(busy));
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };
  if (!response.ok) {
    const message = body.detail || body.message || `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return body;
}

function emptyLoopPayload() {
  return {
    timeline: { rows: [], final_decision: null, last_error: null },
    summary: {},
    trace: {},
  };
}

function objectRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

function safeText(value, fallback = "", maxLength = 120) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return fallback;
  }
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function safeField(value, maxLength = 4000) {
  return String(value || "").slice(0, maxLength);
}

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : 0;
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm;
}

function elapsedSeconds(startedAt) {
  const started = Number(startedAt);
  if (!Number.isFinite(started) || started <= 0) {
    return 0;
  }
  return Math.max(0, Math.floor((Date.now() - started) / 1000));
}

function activeRecipeName() {
  const recipe = state.recipes.find(
    (item) => item.recipe_id === state.activeRecipeId,
  );
  return recipe?.name || "General assistant loop";
}

function selectedEvidenceLabel() {
  const value = String(elements.queryContext?.value || "smart");
  const option = Array.from(elements.queryContext?.children || []).find(
    (item) => item.value === value,
  );
  if (option?.textContent) {
    return option.textContent;
  }
  return value && value !== "smart" ? value : "Automatic";
}

function pendingAssistantMessage({
  contextLabel,
  recipeName,
  queryId,
  userContent,
  baseMessageCount,
}) {
  const startedAt = Date.now();
  return {
    role: "assistant",
    content: "",
    pending: true,
    pending_id: `pending_${startedAt}_${Math.random().toString(36).slice(2, 8)}`,
    local_query_id: queryId,
    user_content: userContent,
    base_message_count: nonNegativeInteger(baseMessageCount),
    started_at: startedAt,
    context_label: contextLabel,
    recipe_name: recipeName,
  };
}

function pendingMessageContent(message) {
  const seconds = elapsedSeconds(message.started_at);
  return [
    "**Thinking...**",
    "",
    `${seconds}s elapsed. Choosing context, drafting an answer, and checking the result.`,
    `Evidence: ${message.context_label || "Automatic"}.`,
  ].join("\n");
}

function runningLoopPayload(message) {
  const seconds = elapsedSeconds(message.started_at);
  return {
    timeline: {
      rows: [
        {
          index: 1,
          phase: "Input",
          phase_key: "input",
          step: "Apply loop recipe",
          decision: "continue",
          signals: message.recipe_name || "General assistant loop",
        },
        {
          index: 2,
          phase: "Context",
          phase_key: "context_select",
          step: "Choose context",
          decision: "continue",
          signals: message.context_label || "Automatic",
        },
        {
          index: 3,
          phase: "Draft",
          phase_key: "draft",
          step: "Waiting for model",
          decision: "running",
          signals: `${seconds}s elapsed; local models can take longer with web evidence or large context`,
        },
        {
          index: 4,
          phase: "Check",
          phase_key: "verify",
          step: "Format and verify answer",
          decision: "pending",
          signals: "will update when the loop returns",
        },
      ],
      final_decision: "running",
      last_error: null,
    },
    summary: {
      status: "running",
      context_provider: message.context_label || "Automatic",
      elapsed_seconds: seconds,
    },
    trace: {
      model_thinking: {
        available: false,
        redacted: false,
        label: "Model Thinking",
        note: "Model thinking appears only after the model response is returned.",
      },
    },
  };
}

function queryErrorLoopPayload(message) {
  const visibleMessage = safeText(message, "request failed", 240);
  return {
    timeline: {
      rows: [
        {
          index: 1,
          phase: "Error",
          phase_key: "error",
          step: "Query failed",
          decision: "error",
          signals: visibleMessage,
        },
      ],
      final_decision: "error",
      last_error: visibleMessage,
    },
    summary: {
      status: "error",
      last_error: visibleMessage,
    },
    trace: {
      error: "query_failed",
    },
  };
}

function removePendingMessage(thread, pendingId) {
  if (!thread || !pendingId) {
    return false;
  }
  const originalLength = thread.messages.length;
  thread.messages = thread.messages.filter(
    (message) => message.pending_id !== pendingId,
  );
  return thread.messages.length !== originalLength;
}

function startQueryProgress(threadId, pendingId) {
  stopQueryProgress();
  state.runningQuery = {
    threadId,
    pendingId,
    timerId: globalThis.setInterval(() => {
      const thread = threadById(threadId);
      const pending = thread?.messages.find(
        (message) => message.pending_id === pendingId,
      );
      if (!pending) {
        stopQueryProgress();
        return;
      }
      if (state.activeThreadId === threadId) {
        renderMessages();
        renderLoopPayload(runningLoopPayload(pending));
      }
    }, QUERY_PROGRESS_INTERVAL_MS),
  };
}

function stopQueryProgress(pendingId = null) {
  if (!state.runningQuery) {
    return;
  }
  if (pendingId && state.runningQuery.pendingId !== pendingId) {
    return;
  }
  globalThis.clearInterval(state.runningQuery.timerId);
  state.runningQuery = null;
}

function preserveLocalInFlightThreadState(serverThread, existingThread) {
  if (!existingThread || state.runningQuery?.threadId !== serverThread.id) {
    return serverThread;
  }
  const pending = existingThread.messages.find(
    (message) => message.pending_id === state.runningQuery.pendingId,
  );
  if (!pending) {
    return serverThread;
  }
  const baseMessageCount = nonNegativeInteger(pending.base_message_count);
  const serverMessageCount = nonNegativeInteger(serverThread.messageCount);
  const serverHasPendingUser = serverMessageCount > baseMessageCount;
  const serverHasPendingAnswer = serverMessageCount >= baseMessageCount + 2;
  const localUserMessages = existingThread.messages.filter(
    (message) =>
      message.role === "user" &&
      message.local_query_id &&
      message.local_query_id === pending.local_query_id,
  );
  const serverMessages = serverThread.messages.filter(
    (message) => !message.pending_id,
  );
  const mergedMessages = [...serverMessages];
  if (!serverHasPendingUser) {
    for (const localMessage of localUserMessages) {
      mergedMessages.push(localMessage);
    }
  }
  if (!serverHasPendingAnswer) {
    mergedMessages.push(pending);
  }
  return {
    ...serverThread,
    messages: mergedMessages.slice(-MAX_THREAD_MESSAGES),
    revision: normalizedRevision(existingThread.revision),
    messageCount: Math.max(existingThread.messageCount, serverThread.messageCount),
    updatedAt: existingThread.updatedAt || serverThread.updatedAt,
  };
}

function runningPayloadForThread(threadId) {
  if (state.runningQuery?.threadId !== threadId) {
    return null;
  }
  const pending = threadById(threadId)?.messages.find(
    (message) => message.pending_id === state.runningQuery.pendingId,
  );
  return pending ? runningLoopPayload(pending) : null;
}

function sanitizeMessage(message) {
  const role = message?.role === "assistant" ? "assistant" : "user";
  const sanitized = {
    role,
    content: String(message?.content || ""),
  };
  if (role === "assistant" && message?.thinking && typeof message.thinking === "object") {
    sanitized.thinking = message.thinking;
  }
  return sanitized;
}

function quarantinedLoopRun(runId, quarantineReason = DEFAULT_QUARANTINE_REASON) {
  return {
    run_id: runId,
    projection_status: "quarantined",
    quarantine_reason: quarantineReason,
    final_decision: "quarantined",
    context_provider: "unavailable",
    backend: "unavailable",
    model: "unavailable",
    recipe_id: "",
    recipe_name: "Stored loop run",
    step_count: 0,
    started_at: "",
    completed_at: "",
    created_at: "",
  };
}

function sanitizeLoopRun(rawRun, expectedThreadId) {
  const runData = objectRecord(rawRun);
  const rawRunId = runData.run_id;
  const runId =
    typeof rawRunId === "string" && SESSION_ID_PATTERN.test(rawRunId)
      ? rawRunId
      : "";
  const rawProjectionStatus = runData.projection_status;
  if (!LOOP_RUN_PROJECTION_STATUSES.has(rawProjectionStatus)) {
    return quarantinedLoopRun(runId);
  }
  if (rawProjectionStatus === "quarantined") {
    try {
      exactPublicObject(runData, QUARANTINED_LOOP_RUN_SUMMARY_KEYS);
      const threadId = publicString(runData.thread_id);
      const quarantineReason = publicEnum(
        runData.quarantine_reason,
        new Set(LOOP_RUN_QUARANTINE_REASON_LABELS.keys()),
      );
      if (
        !SESSION_ID_PATTERN.test(threadId) ||
        threadId !== expectedThreadId ||
        !runId
      ) {
        malformedPublicRunDetail();
      }
      publicTimestamp(runData.created_at);
      return quarantinedLoopRun(runId, quarantineReason);
    } catch {
      return quarantinedLoopRun(runId);
    }
  }

  try {
    exactPublicObject(runData, AVAILABLE_LOOP_RUN_SUMMARY_KEYS);
    const threadId = publicString(runData.thread_id);
    const finalDecision = publicEnum(
      runData.final_decision,
      PUBLIC_FINAL_DECISIONS,
      { nullable: true },
    );
    const terminalReason = publicEnum(
      runData.terminal_reason,
      PUBLIC_TERMINAL_REASONS,
      { nullable: true },
    );
    const allowedReasons = PUBLIC_FINAL_REASON_CONTRACT.get(finalDecision);
    const contextProvider = publicEnum(
      runData.context_provider,
      PUBLIC_CONTEXT_PROVIDERS,
    );
    const backend = publicEnum(runData.backend, PUBLIC_BACKENDS, {
      nullable: true,
    });
    const model = publicDisplayLabel(runData.model, { nullable: true });
    const stepCount = publicInteger(runData.step_count);
    const startedAt = publicTimestamp(runData.started_at);
    const completedAt = publicTimestamp(runData.completed_at, {
      nullable: true,
    });
    const createdAt = publicTimestamp(runData.created_at);
    if (
      !runId ||
      !SESSION_ID_PATTERN.test(threadId) ||
      threadId !== expectedThreadId ||
      runData.projection_schema_version !== PUBLIC_REPORT_PROJECTION_SCHEMA ||
      !allowedReasons?.has(terminalReason) ||
      (PUBLIC_TERMINAL_DECISIONS.has(finalDecision)
        ? backend !== null || model !== null
        : backend === null || model === null)
    ) {
      malformedPublicRunDetail();
    }
    return {
      run_id: runId,
      thread_id: threadId,
      projection_status: "available",
      projection_schema_version: PUBLIC_REPORT_PROJECTION_SCHEMA,
      final_decision: finalDecision,
      terminal_reason: terminalReason,
      context_provider: contextProvider,
      backend,
      model,
      recipe_id: "",
      recipe_name: "General assistant loop",
      step_count: stepCount,
      started_at: startedAt,
      completed_at: completedAt,
      created_at: createdAt,
    };
  } catch {
    return quarantinedLoopRun(runId);
  }
}

function sanitizeRecipe(rawRecipe) {
  const recipeId = String(rawRecipe?.recipe_id || "");
  const detailLoaded = Boolean(
    rawRecipe &&
      (
        Object.hasOwn(rawRecipe, "instructions") ||
        Object.hasOwn(rawRecipe, "success_criteria") ||
        Object.hasOwn(rawRecipe, "stop_condition") ||
        Object.hasOwn(rawRecipe, "metadata") ||
        Object.hasOwn(rawRecipe, "created_at")
      ),
  );
  const successCriteria = Array.isArray(rawRecipe?.success_criteria)
    ? rawRecipe.success_criteria
        .map((criterion) => safeField(criterion, 500).replace(/\s+/g, " ").trim())
        .filter(Boolean)
    : [];
  return {
    recipe_id: SESSION_ID_PATTERN.test(recipeId) ? recipeId : "",
    name: safeText(rawRecipe?.name, "Loop recipe", 96),
    description: safeField(rawRecipe?.description, 500).replace(/\s+/g, " ").trim(),
    goal: safeField(rawRecipe?.goal, 2000),
    instructions: safeField(rawRecipe?.instructions, 4000),
    success_criteria: successCriteria,
    stop_condition: safeField(rawRecipe?.stop_condition, 1000),
    context_provider: String(rawRecipe?.context_provider || "smart"),
    model_profile: String(rawRecipe?.model_profile || "quality"),
    verifier: String(rawRecipe?.verifier || "default"),
    metadata:
      rawRecipe?.metadata && typeof rawRecipe.metadata === "object"
        ? rawRecipe.metadata
        : {},
    detail_loaded: detailLoaded,
    is_default: Boolean(rawRecipe?.is_default),
    created_at: String(rawRecipe?.created_at || ""),
    updated_at: String(rawRecipe?.updated_at || ""),
  };
}

function normalizedRevision(value) {
  return nonNegativeInteger(value);
}

function bumpThreadRevision(thread) {
  thread.revision = normalizedRevision(thread.revision) + 1;
  return thread.revision;
}

function threadMutationInProgress(threadId) {
  return (
    state.deletingThreadTokens.has(threadId) ||
    state.clearingThreadTokens.has(threadId) ||
    state.uploadThreadTokens.has(threadId) ||
    state.runningQuery?.threadId === threadId
  );
}

function ownsThreadOperation(tokens, threadId, token) {
  return tokens.get(threadId) === token;
}

function sanitizeThread(rawThread) {
  const rawId = String(rawThread?.id || "");
  const id = SESSION_ID_PATTERN.test(rawId) ? rawId : "";
  const messages = Array.isArray(rawThread?.messages)
    ? rawThread.messages.slice(-MAX_THREAD_MESSAGES).map(sanitizeMessage)
    : [];
  const latest =
    rawThread?.latest && typeof rawThread.latest === "object" ? rawThread.latest : null;
  const loopRuns = Array.isArray(rawThread?.loop_runs)
    ? rawThread.loop_runs
        .map((run) => sanitizeLoopRun(run, id))
        .filter((run) => run.run_id)
    : [];
  const rawLoopRunCount = Number(rawThread?.loop_run_count ?? rawThread?.loopRunCount);
  const rawMemoryCount = Number(rawThread?.memory_count ?? rawThread?.memoryCount);
  const loopRunCount =
    Number.isSafeInteger(rawLoopRunCount) && rawLoopRunCount >= 0
      ? rawLoopRunCount
      : loopRuns.length;
  const memoryCount =
    Number.isSafeInteger(rawMemoryCount) && rawMemoryCount >= 0
      ? rawMemoryCount
      : 0;
  const createdAt = String(
    rawThread?.created_at || rawThread?.createdAt || new Date().toISOString(),
  );
  const updatedAt = String(
    rawThread?.updated_at || rawThread?.updatedAt || createdAt,
  );
  const messageCount = Number(rawThread?.message_count);
  return {
    id,
    title: safeText(rawThread?.title, DEFAULT_THREAD_TITLE, 64),
    messages,
    loopRuns,
    loopRunCount,
    latest,
    revision: normalizedRevision(rawThread?.revision),
    createdAt,
    updatedAt,
    messageCount: Number.isSafeInteger(messageCount) && messageCount >= 0
      ? messageCount
      : messages.length,
    memoryCount,
  };
}

function applyThreadSummary(thread, summary) {
  if (!thread || !summary?.id || thread.id !== summary.id) {
    return;
  }
  thread.title = summary.title || thread.title;
  thread.updatedAt = summary.updatedAt || thread.updatedAt;
  thread.messageCount = Number.isSafeInteger(summary.messageCount)
    ? summary.messageCount
    : thread.messageCount;
  thread.loopRunCount = Number.isSafeInteger(summary.loopRunCount)
    ? summary.loopRunCount
    : thread.loopRunCount;
  thread.memoryCount = Number.isSafeInteger(summary.memoryCount)
    ? summary.memoryCount
    : thread.memoryCount;
}

function legacyActiveThreadId() {
  try {
    const payload = JSON.parse(
      globalThis.localStorage?.getItem(LEGACY_THREAD_STORAGE_KEY) || "null",
    );
    return String(payload?.activeThreadId || "");
  } catch {
    return "";
  }
}

function loadActiveThreadId() {
  const activeId = String(
    globalThis.localStorage?.getItem(ACTIVE_THREAD_STORAGE_KEY) ||
      globalThis.localStorage?.getItem(LEGACY_ACTIVE_THREAD_STORAGE_KEY) ||
      "",
  );
  return activeId || legacyActiveThreadId();
}

function loadActiveRecipeId() {
  return String(
    globalThis.localStorage?.getItem(ACTIVE_RECIPE_STORAGE_KEY) ||
      globalThis.localStorage?.getItem(LEGACY_ACTIVE_RECIPE_STORAGE_KEY) ||
      "",
  );
}

function persistActiveThreadId() {
  try {
    if (state.activeThreadId) {
      globalThis.localStorage?.setItem(
        ACTIVE_THREAD_STORAGE_KEY,
        state.activeThreadId,
      );
    }
  } catch {
    // Browser storage is only a selected-thread hint. Server threads remain authoritative.
  }
}

function persistActiveRecipeId() {
  try {
    if (state.activeRecipeId) {
      globalThis.localStorage?.setItem(
        ACTIVE_RECIPE_STORAGE_KEY,
        state.activeRecipeId,
      );
    }
  } catch {
    // Browser storage is only a selected-recipe hint. Server recipes remain authoritative.
  }
}

function upsertThread(thread, { moveToTop = false } = {}) {
  if (!thread.id) {
    return;
  }
  const others = state.threads.filter((item) => item.id !== thread.id);
  state.threads = moveToTop
    ? [thread, ...others].slice(0, MAX_THREADS)
    : [thread, ...others].sort((left, right) =>
        String(right.updatedAt).localeCompare(String(left.updatedAt)),
      ).slice(0, MAX_THREADS);
}

async function fetchThreadDetail(threadId) {
  const existingThread = threadById(threadId);
  const detail = sanitizeThread(
    await requestJson(`/api/threads/${encodeURIComponent(threadId)}`),
  );
  if (detail.id !== threadId) {
    throw new Error("Thread detail identity mismatch.");
  }
  return preserveLocalInFlightThreadState(detail, existingThread);
}

async function loadThreadDetail(threadId) {
  const merged = await fetchThreadDetail(threadId);
  upsertThread(merged);
  return merged;
}

async function reconcileThreadDetail({
  threadId,
  expectedRevision,
  tokens,
  token,
}) {
  const current = threadById(threadId);
  if (
    !ownsThreadOperation(tokens, threadId, token) ||
    !current ||
    normalizedRevision(current.revision) !== expectedRevision
  ) {
    return false;
  }
  const detail = await fetchThreadDetail(threadId);
  const latest = threadById(threadId);
  if (
    !ownsThreadOperation(tokens, threadId, token) ||
    !latest ||
    normalizedRevision(latest.revision) !== expectedRevision
  ) {
    return false;
  }
  detail.revision = expectedRevision + 1;
  upsertThread(detail);
  return true;
}

async function loadThreads() {
  const payload = await requestJson("/api/threads");
  state.threads = Array.isArray(payload?.threads)
    ? payload.threads.map(sanitizeThread).filter((thread) => thread.id)
    : [];

  if (!state.threads.length) {
    const created = sanitizeThread(
      await requestJson("/api/threads", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title: DEFAULT_THREAD_TITLE }),
      }),
    );
    state.threads = [created];
  }

  const activeId = loadActiveThreadId();
  state.threadSwitchRequestId += 1;
  state.activeThreadId = state.threads.some((thread) => thread.id === activeId)
    ? activeId
    : state.threads[0].id;
  await loadThreadDetail(state.activeThreadId);
  persistActiveThreadId();
}

function activeThread() {
  let thread = state.threads.find((item) => item.id === state.activeThreadId);
  if (!thread) {
    thread = state.threads[0] || {
      id: "",
      title: DEFAULT_THREAD_TITLE,
      messages: [],
      loopRuns: [],
      loopRunCount: 0,
      memoryCount: 0,
      latest: null,
      revision: 0,
      createdAt: "",
      updatedAt: "",
      messageCount: 0,
    };
    state.threadSwitchRequestId += 1;
    state.activeThreadId = thread.id;
    persistActiveThreadId();
  }
  return thread;
}

function threadById(threadId) {
  return state.threads.find((thread) => thread.id === threadId) || null;
}

function touchThread(thread) {
  thread.updatedAt = new Date().toISOString();
  thread.messages = thread.messages.slice(-MAX_THREAD_MESSAGES);
  thread.messageCount = thread.messages.length;
  upsertThread(thread, { moveToTop: true });
  persistActiveThreadId();
}

function titleFromMessage(message) {
  return safeText(message, DEFAULT_THREAD_TITLE, 54);
}

function renderThreads() {
  elements.threadList.replaceChildren();
  const activeThreadMutating =
    !state.activeThreadId || threadMutationInProgress(state.activeThreadId);
  elements.deleteThreadButton.disabled =
    !state.threads.length ||
    activeThreadMutating;
  elements.clearButton.disabled = activeThreadMutating;
  elements.uploadButton.disabled = activeThreadMutating;
  for (const thread of state.threads) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread-button";
    button.dataset.active = String(thread.id === state.activeThreadId);
    button.disabled =
      state.deletingThreadTokens.has(thread.id) ||
      state.clearingThreadTokens.has(thread.id) ||
      state.uploadThreadTokens.has(thread.id);
    button.addEventListener("click", () =>
      switchThread(thread.id).catch((error) => {
        elements.uploadStatus.textContent = error.message;
      }),
    );

    const title = document.createElement("span");
    title.textContent = thread.title || DEFAULT_THREAD_TITLE;
    const meta = document.createElement("small");
    const messageCount = Number.isSafeInteger(thread.messageCount)
      ? thread.messageCount
      : thread.messages.length;
    const runCount = Number.isSafeInteger(thread.loopRunCount)
      ? thread.loopRunCount
      : Array.isArray(thread.loopRuns) ? thread.loopRuns.length : 0;
    const memoryCount = Number.isSafeInteger(thread.memoryCount)
      ? thread.memoryCount
      : 0;
    meta.textContent = `${messageCount} ${
      messageCount === 1 ? "message" : "messages"
    } · ${runCount} ${runCount === 1 ? "run" : "runs"} · ${memoryCount} ${
      memoryCount === 1 ? "memory" : "memories"
    }`;

    button.append(title, meta);
    elements.threadList.append(button);
  }
}

function memoryCountLabel(count) {
  const value = nonNegativeInteger(count);
  return `${value} ${plural(value, "memory", "memories")} indexed`;
}

function runMemoryLabel(summary) {
  if (!summary || typeof summary !== "object" || !Object.keys(summary).length) {
    return "no completed run yet";
  }
  const data = summary;
  if (
    !Number.isSafeInteger(data.conversation_context_count) ||
    data.conversation_context_count < 0 ||
    !Number.isSafeInteger(data.semantic_memory_count) ||
    data.semantic_memory_count < 0 ||
    typeof data.semantic_memory_status !== "string"
  ) {
    return "last run memory use unknown";
  }
  const recentCount = data.conversation_context_count;
  const semanticCount = data.semantic_memory_count;
  const semanticStatus = data.semantic_memory_status;
  const parts = [];
  if (recentCount) {
    parts.push(`${recentCount} recent ${plural(recentCount, "turn")}`);
  }
  if (semanticStatus === "retrieved" && semanticCount) {
    parts.push(`${semanticCount} recalled ${plural(semanticCount, "memory", "memories")}`);
  }
  if (parts.length) {
    return `last run used ${parts.join(" + ")}`;
  }
  if (semanticStatus === "retrieved") {
    return "last run found no matching memory";
  }
  if (semanticStatus === "empty") {
    return "last run found no matching memory";
  }
  if (semanticStatus === "unavailable") {
    return "last run memory unavailable";
  }
  return "last run did not use thread memory";
}

function renderActiveThreadTitle() {
  const thread = activeThread();
  elements.activeThreadTitle.textContent = thread.title || DEFAULT_THREAD_TITLE;
  elements.activeThreadMemory.textContent = `${memoryCountLabel(
    thread.memoryCount,
  )} · ${runMemoryLabel(thread.latest?.summary)}`;
}

function renderMemoryStatus(payload) {
  const thread = activeThread();
  const summary = payload?.summary || {};
  elements.memoryStatus.replaceChildren();

  const label = document.createElement("span");
  label.textContent = "Thread memory";
  const indexed = document.createElement("strong");
  indexed.textContent = memoryCountLabel(thread.memoryCount);
  const lastRun = document.createElement("span");
  lastRun.textContent = runMemoryLabel(summary);

  elements.memoryStatus.append(label, indexed, lastRun);
}

async function switchThread(threadId) {
  if (!threadById(threadId) || threadId === state.activeThreadId) {
    return;
  }
  const sourceThreadId = state.activeThreadId;
  const requestId = state.threadSwitchRequestId + 1;
  state.threadSwitchRequestId = requestId;
  let detail;
  try {
    detail = await fetchThreadDetail(threadId);
  } catch (error) {
    if (
      state.threadSwitchRequestId === requestId &&
      state.activeThreadId === sourceThreadId
    ) {
      throw error;
    }
    return;
  }
  if (
    state.threadSwitchRequestId !== requestId ||
    state.activeThreadId !== sourceThreadId ||
    !threadById(threadId) ||
    state.deletingThreadTokens.has(threadId) ||
    state.clearingThreadTokens.has(threadId) ||
    state.uploadThreadTokens.has(threadId)
  ) {
    return;
  }
  upsertThread(detail);
  state.activeThreadId = threadId;
  resetRunInspection();
  persistActiveThreadId();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(activeThread().loopRuns);
  renderLoopPayload(
    runningPayloadForThread(threadId) || activeThread().latest || emptyLoopPayload(),
  );
  await refreshStatus().catch((error) => {
    elements.uploadStatus.textContent = error.message;
  });
}

async function startNewThread() {
  const thread = sanitizeThread(
    await requestJson("/api/threads", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: DEFAULT_THREAD_TITLE }),
    }),
  );
  upsertThread(thread, { moveToTop: true });
  state.threadSwitchRequestId += 1;
  state.activeThreadId = thread.id;
  resetRunInspection();
  persistActiveThreadId();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(thread.loopRuns);
  renderLoopPayload(emptyLoopPayload());
  await refreshStatus().catch((error) => {
    elements.uploadStatus.textContent = error.message;
  });
  elements.queryInput.focus();
}

async function deleteActiveThread() {
  const thread = activeThread();
  const threadId = thread.id;
  if (!threadId || threadMutationInProgress(threadId)) {
    return;
  }
  const title = thread.title || DEFAULT_THREAD_TITLE;
  const confirmed = typeof globalThis.confirm === "function"
    ? globalThis.confirm(
        `Delete thread "${title}"? This removes its messages, memories, and runs from this local database.`,
      )
    : true;
  if (!confirmed) {
    return;
  }

  const deleteRevision = normalizedRevision(thread.revision);
  const operationToken = state.threadOperationRequestId + 1;
  state.threadOperationRequestId = operationToken;
  state.deletingThreadTokens.set(threadId, operationToken);
  resetRunInspection();
  setBusy(elements.deleteThreadButton, true, "Delete");
  if (state.activeThreadId === threadId) {
    setQueryControlsBusy(true);
  }
  renderThreads();
  try {
    try {
      await requestJson(`/api/threads/${encodeURIComponent(threadId)}`, {
        method: "DELETE",
      });
    } catch (error) {
      try {
        await reconcileThreadDetail({
          threadId,
          expectedRevision: deleteRevision,
          tokens: state.deletingThreadTokens,
          token: operationToken,
        });
      } catch {
        // The delete did not mutate local state, so the existing snapshot is safer.
      }
      if (
        ownsThreadOperation(
          state.deletingThreadTokens,
          threadId,
          operationToken,
        ) &&
        state.activeThreadId === threadId
      ) {
        renderActiveThreadTitle();
        renderMessages();
        renderRuns(activeThread().loopRuns);
        renderLoopPayload(activeThread().latest || emptyLoopPayload());
        elements.uploadStatus.textContent = error.message;
      }
      return;
    }

    if (
      !ownsThreadOperation(
        state.deletingThreadTokens,
        threadId,
        operationToken,
      )
    ) {
      return;
    }
    state.threads = state.threads.filter((item) => item.id !== threadId);
    if (state.activeThreadId === threadId) {
      state.threadSwitchRequestId += 1;
      state.activeThreadId = state.threads[0]?.id || null;
    }
    if (state.activeThreadId) {
      persistActiveThreadId();
    }
    resetRunInspection();
    renderThreads();
    renderActiveThreadTitle();
    renderMessages();
    renderRuns(activeThread().loopRuns);
    renderLoopPayload(activeThread().latest || emptyLoopPayload());

    let followupError = null;
    try {
      if (!state.threads.length) {
        await loadThreads();
      } else if (
        state.activeThreadId &&
        !state.deletingThreadTokens.has(state.activeThreadId) &&
        !state.clearingThreadTokens.has(state.activeThreadId) &&
        !state.uploadThreadTokens.has(state.activeThreadId)
      ) {
        await loadThreadDetail(state.activeThreadId);
      }
      if (state.activeThreadId) {
        persistActiveThreadId();
      }
      resetRunInspection();
      renderThreads();
      renderActiveThreadTitle();
      renderMessages();
      renderRuns(activeThread().loopRuns);
      renderLoopPayload(activeThread().latest || emptyLoopPayload());
      await refreshStatus();
    } catch (error) {
      followupError = error;
    }
    elements.uploadStatus.textContent = followupError
      ? `Deleted thread "${title}", but could not refresh the replacement: ${followupError.message}`
      : `Deleted thread "${title}".`;
  } finally {
    if (
      ownsThreadOperation(
        state.deletingThreadTokens,
        threadId,
        operationToken,
      )
    ) {
      state.deletingThreadTokens.delete(threadId);
    }
    setBusy(elements.deleteThreadButton, false, "Delete");
    if (
      !state.runningQuery &&
      !state.deletingThreadTokens.has(state.activeThreadId) &&
      !state.clearingThreadTokens.has(state.activeThreadId) &&
      !state.uploadThreadTokens.has(state.activeThreadId)
    ) {
      setQueryControlsBusy(false);
    }
    renderThreads();
  }
}

function sessionFileMessage(status) {
  const fileName = safeText(status?.active_document, "", 96);
  const attemptedName = safeText(status?.last_attempted_document, "", 96);
  const lastError = safeText(status?.last_error, "", 180);
  if (lastError && attemptedName) {
    return fileName
      ? `Could not attach ${attemptedName}. ${fileName} remains attached to this session.`
      : `Could not attach ${attemptedName}. No file is attached to this session.`;
  }
  if (fileName) {
    return `${fileName} is attached to this session.`;
  }
  return "No file attached to this session.";
}

function renderRuntimeStatus(status, { uploadMessage = "" } = {}) {
  const fileName = safeText(status?.active_document, "", 96);
  elements.backendPill.textContent = safeText(status?.backend, "backend", 48);
  elements.modelPill.textContent = safeText(status?.model, "model unavailable", 72);
  elements.readyPill.textContent = fileName ? "file attached" : "no file";
  elements.readyPill.dataset.ready = String(Boolean(fileName));
  elements.fileScope.textContent = fileName || "No file attached";
  elements.uploadStatus.textContent = uploadMessage || sessionFileMessage(status);
}

function uploadStillMatchesActiveThread(threadId, operationToken) {
  return (
    state.activeThreadId === threadId &&
    ownsThreadOperation(
      state.uploadThreadTokens,
      threadId,
      operationToken,
    )
  );
}

function renderMessages() {
  const messages = activeThread().messages;
  elements.messages.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No messages yet.";
    elements.messages.append(empty);
    return;
  }

  for (const message of messages) {
    const bubble = document.createElement("article");
    bubble.className = `message ${message.role}${message.pending ? " pending" : ""}`;
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "You" : "Loop";
    const content = renderMessageContent(
      message.pending ? pendingMessageContent(message) : message.content,
    );
    bubble.append(role, content);
    const thinking = renderMessageThinking(message.thinking);
    if (thinking) {
      bubble.append(thinking);
    }
    elements.messages.append(bubble);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function appendTextWithLineBreaks(parent, text) {
  const lines = String(text || "").split("\n");
  for (const [index, line] of lines.entries()) {
    if (line) {
      const span = document.createElement("span");
      span.textContent = line;
      parent.append(span);
    }
    if (index < lines.length - 1) {
      parent.append(document.createElement("br"));
    }
  }
}

function normalizeMessageMarkdownStructure(text) {
  const compactLabeledListPattern =
    /(?:^|\s)1[.)]\s+(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?/;
  const compactNumberedListPattern =
    /(?:^|:)\s*1[.)]\s+\S[^\n]{2,}?\s+2[.)]\s+\S/s;
  const listItemPattern = /(^|\s+)(\d{1,2})([.)])\s+(?=\S)/g;
  const listStartPattern =
    /(?:^|:)\s*1[.)]\s+(?=\S)|(?:^|\s)1[.)]\s+(?=(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?)/;
  const splitSequentialListItems = (line) => {
    const startMatch = line.match(listStartPattern);
    if (!startMatch || startMatch.index === undefined) {
      return line;
    }
    const markerOffset = startMatch[0].search(/1[.)]\s+/);
    if (markerOffset < 0) {
      return line;
    }
    const listStart = startMatch.index + markerOffset;
    const prefix = line.slice(0, listStart);
    const body = line.slice(listStart);
    let expectedNextItem = null;
    const normalizedBody = body.replace(
      listItemPattern,
      (marker, spacing, rawNumber, punctuation, offset) => {
        const number = Number.parseInt(rawNumber, 10);
        const shouldStartList = number === 1;
        const shouldContinueList = expectedNextItem === number;
        if (shouldStartList) {
          expectedNextItem = 2;
        } else if (shouldContinueList) {
          expectedNextItem = number + 1;
        } else {
          return marker;
        }
        const separator = offset > 0 && !spacing.includes("\n") ? "\n" : spacing;
        return `${separator}${rawNumber}${punctuation} `;
      },
    );
    const listSeparator = prefix && !prefix.endsWith("\n") ? "\n" : "";
    return `${prefix}${listSeparator}${normalizedBody}`;
  };
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split(/(`[^`\n]+`)/g)
    .map((part) =>
      part.startsWith("`")
        ? part
        : compactLabeledListPattern.test(part) ||
            compactNumberedListPattern.test(part)
          ? part.split("\n").map(splitSequentialListItems).join("\n")
          : part,
    )
    .join("");
}

function appendInlineMarkdown(parent, text) {
  const inlinePattern = /`([^`\n]+)`|\*\*([^\n]+?)\*\*/g;
  let cursor = 0;
  for (const match of text.matchAll(inlinePattern)) {
    if (match.index > cursor) {
      appendTextWithLineBreaks(parent, text.slice(cursor, match.index));
    }
    if (match[1] !== undefined) {
      const code = document.createElement("code");
      code.className = "message-inline-code";
      code.textContent = match[1];
      parent.append(code);
    } else {
      const strong = document.createElement("strong");
      strong.textContent = match[2];
      parent.append(strong);
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    appendTextWithLineBreaks(parent, text.slice(cursor));
  }
}

function appendParagraph(container, lines) {
  const text = lines.join("\n").trim();
  if (!text) {
    return;
  }
  const paragraph = document.createElement("p");
  paragraph.className = "message-paragraph";
  appendInlineMarkdown(paragraph, text);
  container.append(paragraph);
}

function appendList(container, items, ordered) {
  if (!items.length) {
    return;
  }
  const list = document.createElement(ordered ? "ol" : "ul");
  list.className = "message-list";
  for (const item of items) {
    const element = document.createElement("li");
    appendInlineMarkdown(element, item);
    list.append(element);
  }
  container.append(list);
}

function appendTextBlocks(container, text) {
  const lines = normalizeMessageMarkdownStructure(text).split("\n");
  let paragraph = [];
  let listItems = [];
  let orderedList = false;

  const flushParagraph = () => {
    appendParagraph(container, paragraph);
    paragraph = [];
  };
  const flushList = () => {
    appendList(container, listItems, orderedList);
    listItems = [];
    orderedList = false;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      const nextOrdered = Boolean(orderedMatch);
      if (listItems.length && orderedList !== nextOrdered) {
        flushList();
      }
      orderedList = nextOrdered;
      listItems.push((orderedMatch || unorderedMatch)[1]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
}

function appendCodeBlock(container, language, codeText) {
  const figure = document.createElement("figure");
  figure.className = "message-code-block";

  const normalizedLanguage = String(language || "").trim();
  if (normalizedLanguage) {
    const caption = document.createElement("figcaption");
    caption.className = "message-code-language";
    caption.textContent = normalizedLanguage;
    figure.append(caption);
  }

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = String(codeText || "").replace(/^\n/, "").replace(/\s+$/, "");
  pre.append(code);
  figure.append(pre);
  container.append(figure);
}

function renderMessageContent(text) {
  const container = document.createElement("div");
  container.className = "message-content";
  const source = String(text || "");
  const fencePattern = /```([A-Za-z0-9_+#.-]*)[ \t]*\n?([\s\S]*?)```/g;
  let cursor = 0;

  for (const match of source.matchAll(fencePattern)) {
    if (match.index > cursor) {
      appendTextBlocks(container, source.slice(cursor, match.index));
    }
    appendCodeBlock(container, match[1], match[2]);
    cursor = match.index + match[0].length;
  }

  if (cursor < source.length) {
    appendTextBlocks(container, source.slice(cursor));
  }
  if (!container.children.length) {
    appendParagraph(container, [source]);
  }
  return container;
}

function renderMessageThinking(thinking) {
  const data = thinking || {};
  const hasThinking = Boolean(data.available && data.content);
  const isRedacted = Boolean(data.redacted);
  if (!hasThinking && !isRedacted) {
    return null;
  }

  const details = document.createElement("details");
  details.className = "message-thinking";
  details.open = true;

  const summary = document.createElement("summary");
  const label = document.createElement("span");
  label.textContent = data.label || "Model Thinking (unverified)";
  const state = document.createElement("small");
  state.textContent = isRedacted ? "redacted" : "captured";
  state.dataset.state = isRedacted ? "redacted" : "captured";
  summary.append(label, state);

  const note = document.createElement("p");
  note.className = "message-thinking-note";
  note.textContent =
    data.note ||
    "Model-emitted thinking is useful for debugging the loop, but it is not verified evidence.";

  const content = document.createElement("pre");
  content.textContent = data.content || "[redacted]";

  details.append(summary, note, content);
  return details;
}

function renderTimeline(timeline) {
  elements.timeline.replaceChildren();
  const rows = timeline?.rows || [];
  elements.finalDecision.textContent = timeline?.final_decision || "idle";
  elements.finalDecision.dataset.decision = timeline?.final_decision || "idle";

  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No loop run yet.";
    elements.timeline.append(empty);
    return;
  }

  for (const row of rows) {
    const item = document.createElement("article");
    item.className = "timeline-row";
    item.dataset.phase = row.phase_key || "";

    const index = document.createElement("span");
    index.className = "timeline-index";
    index.textContent = String(row.index);

    const body = document.createElement("div");
    const heading = document.createElement("div");
    heading.className = "timeline-heading";

    const phase = document.createElement("strong");
    phase.textContent = row.phase || "Step";
    const decision = document.createElement("span");
    decision.className = "decision";
    decision.textContent = row.decision || "continue";
    heading.append(phase, decision);

    const name = document.createElement("p");
    name.className = "timeline-name";
    name.textContent = row.step || "-";

    const signals = document.createElement("p");
    signals.className = "timeline-signals";
    signals.textContent = row.signals || "-";

    body.append(heading, name, signals);
    item.append(index, body);
    elements.timeline.append(item);
  }

  if (timeline.last_error) {
    const error = document.createElement("p");
    error.className = "timeline-error";
    error.textContent = `Last error: ${timeline.last_error}`;
    elements.timeline.append(error);
  }
}

function loopPhaseLabel(phase) {
  const labels = {
    input: "Input",
    context_select: "Context",
    retrieve: "Retrieve",
    draft: "Draft",
    format_check: "Format",
    mechanical_check: "Check",
    verify: "Verify",
    retry: "Retry",
    refuse: "Refuse",
    final: "Final",
    error: "Error",
  };
  const phaseKey = String(phase || "step");
  return labels[phaseKey] || phaseKey.replaceAll("_", " ").replace(/\b\w/g, (value) =>
    value.toUpperCase(),
  );
}

function malformedPublicRunDetail() {
  throw new Error("Stored public run detail is unavailable or malformed.");
}

function exactPublicObject(value, allowedKeys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    malformedPublicRunDetail();
  }
  const keys = Object.keys(value);
  if (
    keys.length !== allowedKeys.length ||
    keys.some((key) => !allowedKeys.includes(key))
  ) {
    malformedPublicRunDetail();
  }
  return value;
}

function allowedPublicObject(value, allowedKeys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    malformedPublicRunDetail();
  }
  if (Object.keys(value).some((key) => !allowedKeys.includes(key))) {
    malformedPublicRunDetail();
  }
  return value;
}

function publicString(value, { nullable = false, category = false } = {}) {
  if (nullable && value === null) {
    return null;
  }
  if (
    typeof value !== "string" ||
    (category && !PUBLIC_CATEGORY_PATTERN.test(value))
  ) {
    malformedPublicRunDetail();
  }
  return value;
}

function publicDisplayLabel(value, { nullable = false } = {}) {
  const label = publicString(value, { nullable });
  if (label === null) {
    return null;
  }
  const codePoints = Array.from(label);
  if (
    !codePoints.length ||
    codePoints.every((character) =>
      publicDisplayCodePointIsBlank(character.codePointAt(0)),
    ) ||
    codePoints.length > 512 ||
    label.includes("://") ||
    codePoints.some((character) =>
      publicDisplayCodePointIsControl(character.codePointAt(0)),
    )
  ) {
    malformedPublicRunDetail();
  }
  return label;
}

function publicDisplayCodePointIsBlank(codePoint) {
  return (
    PUBLIC_DISPLAY_BLANK_CODE_POINTS.has(codePoint) ||
    (0x0080 <= codePoint && codePoint <= 0x009f) ||
    PUBLIC_DISPLAY_FORMAT_CODE_POINT_RANGES.some(
      ([start, end]) => start <= codePoint && codePoint <= end,
    ) ||
    PUBLIC_DISPLAY_MARK_CODE_POINT_RANGES.some(
      ([start, end]) => start <= codePoint && codePoint <= end,
    )
  );
}

function publicDisplayCodePointIsControl(codePoint) {
  return codePoint < 32 || (0x007f <= codePoint && codePoint <= 0x009f);
}

function publicBoolean(value) {
  if (typeof value !== "boolean") {
    malformedPublicRunDetail();
  }
  return value;
}

function publicInteger(value, { minimum = 0, nullable = false } = {}) {
  if (nullable && value === null) {
    return null;
  }
  if (!Number.isSafeInteger(value) || value < minimum) {
    malformedPublicRunDetail();
  }
  return value;
}

function publicTimestamp(value, { nullable = false } = {}) {
  const timestamp = publicString(value, { nullable });
  if (timestamp === null) {
    return null;
  }
  if (!timestamp.endsWith("Z") || Number.isNaN(Date.parse(timestamp))) {
    malformedPublicRunDetail();
  }
  return timestamp;
}

function publicEnum(value, allowed, { nullable = false } = {}) {
  if (nullable && value === null) {
    return null;
  }
  if (typeof value !== "string" || !allowed.has(value)) {
    malformedPublicRunDetail();
  }
  return value;
}

function normalizePublicPolicy(value) {
  const policy = exactPublicObject(value, PUBLIC_POLICY_KEYS);
  return {
    max_retries: publicInteger(policy.max_retries),
    require_citations: publicBoolean(policy.require_citations),
    require_verifier_for_supported: publicBoolean(
      policy.require_verifier_for_supported,
    ),
    allow_mock_supported: publicBoolean(policy.allow_mock_supported),
    allow_tool_calls: publicBoolean(policy.allow_tool_calls),
    require_human_review_for_tools: publicBoolean(
      policy.require_human_review_for_tools,
    ),
  };
}

function normalizePublicVerification(value) {
  if (value === null) {
    return null;
  }
  const verification = exactPublicObject(value, PUBLIC_VERIFICATION_KEYS);
  return {
    outcome: publicEnum(
      verification.outcome,
      PUBLIC_VERIFICATION_OUTCOMES,
    ),
    verifier_backend: publicEnum(verification.verifier_backend, PUBLIC_BACKENDS, {
      nullable: true,
    }),
    verifier_model_label: publicDisplayLabel(verification.verifier_model_label, {
      nullable: true,
    }),
    same_model_as_drafter:
      verification.same_model_as_drafter === null
        ? null
        : publicBoolean(verification.same_model_as_drafter),
  };
}

function normalizePublicStep(value) {
  const step = exactPublicObject(value, PUBLIC_STEP_KEYS);
  const stepId = publicString(step.step_id);
  if (!SESSION_ID_PATTERN.test(stepId)) {
    malformedPublicRunDetail();
  }
  return {
    step_id: stepId,
    phase: publicEnum(step.phase, PUBLIC_PHASES),
    decision: publicEnum(step.decision, PUBLIC_DECISIONS),
    started_at: publicTimestamp(step.started_at),
    ended_at: publicTimestamp(step.ended_at, { nullable: true }),
    duration_ms: publicInteger(step.duration_ms, { nullable: true }),
    backend: publicEnum(step.backend, PUBLIC_BACKENDS, { nullable: true }),
    model_label: publicDisplayLabel(step.model_label, { nullable: true }),
    retry_count: publicInteger(step.retry_count),
    error_present: publicBoolean(step.error_present),
    verification: normalizePublicVerification(step.verification),
    human_review_required: publicBoolean(step.human_review_required),
  };
}

function normalizePublicEvidence(value) {
  const evidence = exactPublicObject(value, PUBLIC_EVIDENCE_KEYS);
  const locator = exactPublicObject(
    evidence.locator,
    PUBLIC_EVIDENCE_LOCATOR_KEYS,
  );
  const evidenceId = publicString(evidence.evidence_id);
  if (!PUBLIC_EVIDENCE_ID_PATTERN.test(evidenceId)) {
    malformedPublicRunDetail();
  }
  return {
    evidence_id: evidenceId,
    citation_id: publicInteger(evidence.citation_id, { minimum: 1 }),
    provider: publicEnum(evidence.provider, PUBLIC_EVIDENCE_PROVIDERS),
    locator: {
      page: publicInteger(locator.page, { nullable: true }),
      chunk_index: publicInteger(locator.chunk_index, { nullable: true }),
    },
  };
}

function validatePublicRedactionContract({
  redactionApplied,
  finalDecision,
  terminalReason,
  finalAnswer,
  backend,
  modelLabel,
  contextProvider,
  steps,
  evidence,
}) {
  const allowedReasons = PUBLIC_FINAL_REASON_CONTRACT.get(finalDecision);
  if (!allowedReasons || !allowedReasons.has(terminalReason)) {
    malformedPublicRunDetail();
  }

  const terminalDecision = PUBLIC_TERMINAL_DECISIONS.has(finalDecision);
  if (redactionApplied !== terminalDecision) {
    malformedPublicRunDetail();
  }

  for (const step of steps) {
    const expectedFinalDecisions = new Set();
    if (step.phase === "refuse" || step.decision === "refuse") {
      expectedFinalDecisions.add("refuse");
    }
    if (step.decision === "block") {
      expectedFinalDecisions.add("block");
    }
    if (step.decision === "requires_review" || step.human_review_required) {
      expectedFinalDecisions.add("requires_review");
    }
    if (
      expectedFinalDecisions.size > 1 ||
      [...expectedFinalDecisions].some(
        (expectedFinalDecision) => finalDecision !== expectedFinalDecision,
      )
    ) {
      malformedPublicRunDetail();
    }
  }

  if (
    (contextProvider === "none" && evidence.length > 0) ||
    evidence.some((item) => item.provider !== contextProvider)
  ) {
    malformedPublicRunDetail();
  }

  if (redactionApplied) {
    if (
      finalAnswer !== null ||
      evidence.length > 0 ||
      backend !== null ||
      modelLabel !== null
    ) {
      malformedPublicRunDetail();
    }
    for (const step of steps) {
      const verification = step.verification;
      if (
        step.backend !== null ||
        step.model_label !== null ||
        (verification !== null &&
          (verification.verifier_backend !== null ||
            verification.verifier_model_label !== null ||
            verification.same_model_as_drafter !== null))
      ) {
        malformedPublicRunDetail();
      }
    }
    return;
  }

  if (
    finalAnswer !== null &&
    !["final", "supported", "not_verified"].includes(finalDecision)
  ) {
    malformedPublicRunDetail();
  }
}

function normalizePublicReport(value, expectedThreadId, expectedRunId) {
  const report = exactPublicObject(value, PUBLIC_REPORT_KEYS);
  const redaction = exactPublicObject(
    report.public_redaction,
    PUBLIC_REDACTION_KEYS,
  );
  const run = exactPublicObject(report.run, PUBLIC_RUN_KEYS);
  const runId = publicString(run.run_id);
  const sessionId = publicString(run.session_id);
  if (
    report.schema_version !== PUBLIC_REPORT_SOURCE_SCHEMA ||
    report.projection_schema_version !== PUBLIC_REPORT_PROJECTION_SCHEMA ||
    report.public !== true ||
    runId !== expectedRunId ||
    sessionId !== expectedThreadId ||
    !SESSION_ID_PATTERN.test(runId) ||
    !SESSION_ID_PATTERN.test(sessionId)
  ) {
    malformedPublicRunDetail();
  }
  const redactionApplied = publicBoolean(redaction.applied);
  const redactionReason = publicString(redaction.reason, { nullable: true });
  if (
    (redactionApplied && redactionReason !== PUBLIC_REDACTION_REASON) ||
    (!redactionApplied && redactionReason !== null)
  ) {
    malformedPublicRunDetail();
  }
  if (!Array.isArray(run.steps) || !Array.isArray(run.evidence)) {
    malformedPublicRunDetail();
  }
  const contextProvider = publicEnum(
    run.context_provider,
    PUBLIC_CONTEXT_PROVIDERS,
  );
  const backend = publicEnum(run.backend, PUBLIC_BACKENDS, { nullable: true });
  const modelLabel = publicDisplayLabel(run.model_label, { nullable: true });
  const finalDecision = publicEnum(
    run.final_decision,
    PUBLIC_FINAL_DECISIONS,
    { nullable: true },
  );
  const terminalReason = publicEnum(
    run.terminal_reason,
    PUBLIC_TERMINAL_REASONS,
    { nullable: true },
  );
  const finalAnswer = publicString(run.final_answer, { nullable: true });
  const conversationContextCount = publicInteger(
    run.conversation_context_count,
    { nullable: true },
  );
  const semanticMemoryCount = publicInteger(run.semantic_memory_count, {
    nullable: true,
  });
  const semanticMemoryStatus = publicEnum(
    run.semantic_memory_status,
    PUBLIC_MEMORY_STATUSES,
    { nullable: true },
  );
  const memoryProvenance = [
    conversationContextCount,
    semanticMemoryCount,
    semanticMemoryStatus,
  ];
  const allMemoryProvenanceMissing = memoryProvenance.every(
    (item) => item === null,
  );
  if (
    (!allMemoryProvenanceMissing && memoryProvenance.includes(null)) ||
    (!allMemoryProvenanceMissing &&
      (semanticMemoryStatus === "retrieved") !== (semanticMemoryCount > 0))
  ) {
    malformedPublicRunDetail();
  }
  const steps = run.steps.map(normalizePublicStep);
  const evidence = run.evidence.map(normalizePublicEvidence);
  if (
    new Set(steps.map((step) => step.step_id)).size !== steps.length ||
    new Set(evidence.map((item) => item.evidence_id)).size !== evidence.length ||
    new Set(evidence.map((item) => item.citation_id)).size !== evidence.length
  ) {
    malformedPublicRunDetail();
  }
  validatePublicRedactionContract({
    redactionApplied,
    finalDecision,
    terminalReason,
    finalAnswer,
    backend,
    modelLabel,
    contextProvider,
    steps,
    evidence,
  });
  return {
    schema_version: PUBLIC_REPORT_SOURCE_SCHEMA,
    projection_schema_version: PUBLIC_REPORT_PROJECTION_SCHEMA,
    public: true,
    public_redaction: {
      applied: redactionApplied,
      reason: redactionReason,
    },
    run: {
      run_id: runId,
      session_id: sessionId,
      context_provider: contextProvider,
      conversation_context_count: conversationContextCount,
      semantic_memory_count: semanticMemoryCount,
      semantic_memory_status: semanticMemoryStatus,
      backend,
      model_label: modelLabel,
      policy: normalizePublicPolicy(run.policy),
      started_at: publicTimestamp(run.started_at),
      completed_at: publicTimestamp(run.completed_at, { nullable: true }),
      steps,
      evidence,
      final_decision: finalDecision,
      terminal_reason: terminalReason,
      final_answer: finalAnswer,
      error_present: publicBoolean(run.error_present),
    },
  };
}

function publicStepSignals(step) {
  const verification = objectRecord(step.verification);
  const parts = [];
  if (verification.outcome) {
    parts.push(`verifier: ${verification.outcome}`);
  }
  if (nonNegativeInteger(step.retry_count)) {
    parts.push(`retry #${nonNegativeInteger(step.retry_count)}`);
  }
  if (step.error_present) {
    parts.push("error recorded");
  }
  if (step.human_review_required) {
    parts.push("human review required");
  }
  return safeText(parts.join("; "), "-", 600);
}

function publicEvidenceCitation(evidence) {
  const data = objectRecord(evidence);
  const locator = objectRecord(data.locator);
  return {
    evidence_id: safeText(data.evidence_id, "", 160) || null,
    citation_id: data.citation_id ?? null,
    provider: safeText(data.provider, "unknown", 80),
    locator: {
      page: locator.page ?? null,
      chunk_index: locator.chunk_index ?? null,
    },
  };
}

function publicSelfCheck(selfCheck) {
  const data = objectRecord(selfCheck);
  if (!Object.keys(data).length) {
    return null;
  }
  return {
    outcome: data.outcome || null,
    reasons: Array.isArray(data.reasons)
      ? data.reasons.map((reason) => safeText(reason, "", 400)).filter(Boolean)
      : [],
    retry_attempted: Boolean(data.retry_attempted),
  };
}

function publicCheckOutcome(step) {
  if (!step) {
    return null;
  }
  if (step.error_present || step.decision === "error") {
    return "failed";
  }
  if (step.decision === "retry") {
    return "retry";
  }
  return "passed";
}

function historicalModelThinking(redacted) {
  return {
    available: false,
    redacted: Boolean(redacted),
    label: "Model Thinking (unverified)",
    content: null,
    note: "Model thinking is not included in public durable-run records.",
  };
}

function historicalLoopPayload(detail, expectedThreadId, expectedRunId) {
  const payload = allowedPublicObject(detail, PUBLIC_DETAIL_KEYS);
  const detailRunId = payload.run_id;
  const detailThreadId = payload.thread_id;
  if (
    payload.public !== true ||
    typeof detailRunId !== "string" ||
    typeof detailThreadId !== "string" ||
    detailThreadId !== expectedThreadId ||
    detailRunId !== expectedRunId
  ) {
    malformedPublicRunDetail();
  }
  const report = normalizePublicReport(
    payload.report,
    expectedThreadId,
    expectedRunId,
  );
  const run = report.run;
  const runId = run.run_id;
  const steps = run.steps;
  const evidence = run.evidence;
  const formatSteps = steps.filter((step) => step.phase === "format_check");
  const mechanicalSteps = steps.filter(
    (step) => step.phase === "mechanical_check",
  );
  const verifySteps = steps.filter((step) => step.phase === "verify");
  const verifyStep = verifySteps.at(-1);
  const verification = objectRecord(verifyStep?.verification);
  const publicRedaction = objectRecord(report.public_redaction);
  const finalDecision = run.final_decision || null;
  const ordinaryRefusal =
    publicRedaction.applied === true &&
    finalDecision === "refuse" &&
    ["verification_failed", "retry_budget_exhausted"].includes(
      run.terminal_reason,
    );
  const publicError = publicRedaction.applied === true
    ? ordinaryRefusal
      ? null
      : PUBLIC_REDACTION_REASON
    : run.error_present === true
      ? "loop_error"
      : null;
  const thinkingRedactionApplied = [
    "block",
    "error",
    "refuse",
    "requires_review",
  ].includes(finalDecision) || publicRedaction.applied === true;
  const citations = evidence.map(publicEvidenceCitation);
  const citationStatus = citations.length
    ? "available"
    : publicRedaction.applied === true
      ? "redacted"
      : "unavailable";
  const citationDetails =
    citationStatus === "redacted"
      ? "Citation details redacted for this terminal run."
      : citationStatus === "available"
        ? "Citation identities and locators are available from the stored public report."
        : "No public citation identities are available for this run.";
  const timeline = {
    rows: steps.map((step, index) => ({
      index: index + 1,
      phase: loopPhaseLabel(step.phase),
      phase_key: String(step.phase || ""),
      decision: String(step.decision || "continue"),
      step: safeText(step.name, loopPhaseLabel(step.phase), 160),
      signals: publicStepSignals(step),
    })),
    final_decision: finalDecision,
    terminal_reason: run.terminal_reason || null,
    last_error: publicError,
  };
  const summary = {
    source: "durable_public_report",
    public: true,
    schema_version: report.schema_version || null,
    projection_schema_version: report.projection_schema_version,
    run_id: runId,
    started_at: run.started_at || null,
    completed_at: run.completed_at || null,
    context_provider: run.context_provider || null,
    backend: run.backend || null,
    model: run.model_label || null,
    conversation_context_count: run.conversation_context_count,
    semantic_memory_count: run.semantic_memory_count,
    semantic_memory_status: run.semantic_memory_status,
    recipe_id: null,
    recipe_name: null,
    step_count: steps.length,
    draft_attempt_count: steps.filter((step) => step.phase === "draft").length,
    format_check: publicCheckOutcome(formatSteps.at(-1)),
    mechanical_check: publicCheckOutcome(mechanicalSteps.at(-1)),
    verifier: verifyStep && Object.keys(verification).length
      ? {
          decision: verifyStep.decision || null,
          outcome: verification.outcome || null,
          reasons: [],
        }
      : null,
    retry_attempted: steps.some((step) => step.phase === "retry"),
    refused:
      finalDecision === "refuse" || steps.some((step) => step.phase === "refuse"),
    final_decision: finalDecision,
    terminal_reason: run.terminal_reason || null,
    last_error: publicError,
    citation_details: citationDetails,
    public_redaction: publicRedaction.applied
      ? {
          applied: true,
          reason: publicRedaction.reason || null,
        }
      : { applied: false },
  };
  return {
    payload: {
      timeline,
      summary,
      trace: {
        source: "durable_public_report",
        public: true,
        question: null,
        answer: publicRedaction.applied === true
          ? ordinaryRefusal
            ? SELF_CHECK_REFUSAL_ANSWER
            : PUBLIC_REDACTION_TEXT
          : (run.final_answer ?? null),
        document: null,
        backend: run.backend ?? null,
        model: run.model_label ?? null,
        retrieved_chunk_count: citations.length,
        citations,
        citation_details: citationDetails,
        self_check: verifyStep
          ? publicSelfCheck({
              ...verification,
              retry_attempted: steps.some((step) => step.phase === "retry"),
            })
          : null,
        model_thinking: historicalModelThinking(thinkingRedactionApplied),
        loop_report: report,
        terminal_reason: run.terminal_reason || null,
        error: publicError,
      },
    },
    citationStatus,
  };
}

function resetRunInspection(message = "") {
  state.runInspection = {
    requestId: state.runInspection.requestId + 1,
    threadId: state.activeThreadId,
    runId: null,
    pendingRunId: null,
    status: "idle",
    message,
  };
}

function runInspectorMessage(durableRuns) {
  const inspection = state.runInspection;
  const inspectableRunCount = durableRuns.filter(
    (run) => run.projection_status === "available",
  ).length;
  if (
    inspection.threadId === state.activeThreadId &&
    inspection.message
  ) {
    return inspection.message;
  }
  if (activeThread().latest) {
    return inspectableRunCount
      ? "Showing the latest run. Select a stored run to inspect its public record."
      : durableRuns.length
        ? "Showing the latest run. Stored runs are quarantined and cannot be inspected."
        : "Showing the latest run. No stored public run is available to inspect.";
  }
  if (inspectableRunCount) {
    return "Select a stored run to inspect its public record.";
  }
  if (durableRuns.length) {
    return "Stored runs are quarantined and cannot be inspected.";
  }
  return "No public run record is available to inspect.";
}

function renderRunInspectorStatus(durableRuns) {
  const status = document.createElement("p");
  status.className = "run-inspector-status";
  status.role = "status";
  status.ariaLive = "polite";
  status.dataset.state =
    state.runInspection.threadId === state.activeThreadId
      ? state.runInspection.status
      : "idle";
  status.textContent = runInspectorMessage(durableRuns);
  elements.runList.append(status);
}

function restoreInspectedRunFocus() {
  if (
    state.runInspection.threadId !== state.activeThreadId ||
    !state.runInspection.runId
  ) {
    return;
  }
  const selected = Array.from(elements.runList.children).find(
    (item) =>
      item.className === "run-row" && item.dataset.selected === "true",
  );
  selected?.focus({ preventScroll: true });
}

function runInspectionIsCurrent(requestId, threadId, runId) {
  return (
    state.activeThreadId === threadId &&
    state.runInspection.requestId === requestId &&
    state.runInspection.threadId === threadId &&
    state.runInspection.pendingRunId === runId
  );
}

async function inspectDurableRun(runId) {
  const threadId = state.activeThreadId;
  const selectedRun = activeThread().loopRuns.find(
    (run) => run.run_id === runId,
  );
  if (
    !SESSION_ID_PATTERN.test(String(threadId || "")) ||
    selectedRun?.projection_status !== "available" ||
    state.deletingThreadTokens.has(threadId) ||
    state.clearingThreadTokens.has(threadId) ||
    state.uploadThreadTokens.has(threadId) ||
    state.runningQuery?.threadId === threadId
  ) {
    return;
  }
  const requestId = state.runInspection.requestId + 1;
  state.runInspection = {
    requestId,
    threadId,
    runId:
      state.runInspection.threadId === threadId
        ? state.runInspection.runId
        : null,
    pendingRunId: runId,
    status: "loading",
    message: `Loading stored public run ${runId}…`,
  };
  renderRuns(activeThread().loopRuns);
  try {
    const detail = await requestJson(
      `/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}`,
    );
    if (!runInspectionIsCurrent(requestId, threadId, runId)) {
      return;
    }
    const historical = historicalLoopPayload(detail, threadId, runId);
    state.runInspection.runId = runId;
    state.runInspection.pendingRunId = null;
    state.runInspection.status = "loaded";
    state.runInspection.message =
      historical.citationStatus === "available"
        ? `Viewing stored public run ${runId}.`
        : historical.citationStatus === "redacted"
          ? `Viewing stored public run ${runId}. Citation details are terminal-redacted.`
          : `Viewing stored public run ${runId}. No public citation identities are available.`;
    renderRuns(activeThread().loopRuns);
    renderLoopPayload(historical.payload);
  } catch (error) {
    if (!runInspectionIsCurrent(requestId, threadId, runId)) {
      return;
    }
    state.runInspection.pendingRunId = null;
    state.runInspection.status = "error";
    state.runInspection.message = `Could not inspect ${runId}: ${safeText(
      error.message,
      "request failed",
      240,
    )}`;
    renderRuns(activeThread().loopRuns);
  }
}

function renderRuns(runs) {
  const durableRuns = Array.isArray(runs) ? runs : [];
  elements.runList.replaceChildren();
  elements.runCount.textContent = `${durableRuns.length} stored`;
  if (!durableRuns.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No durable loop runs stored for this thread yet.";
    elements.runList.append(empty);
    renderRunInspectorStatus(durableRuns);
    return;
  }

  for (const run of durableRuns) {
    const quarantined = run.projection_status === "quarantined";
    const row = document.createElement("button");
    row.type = "button";
    row.className = "run-row";
    const isInspected =
      !quarantined &&
      state.runInspection.threadId === state.activeThreadId &&
      state.runInspection.runId === run.run_id;
    const isPending =
      !quarantined &&
      state.runInspection.threadId === state.activeThreadId &&
      state.runInspection.pendingRunId === run.run_id;
    row.dataset.selected = String(isInspected);
    row.dataset.state = quarantined
      ? "quarantined"
      : isPending
        ? "loading"
        : isInspected
          ? "loaded"
          : "idle";
    row.ariaPressed = String(isInspected);
    row.ariaBusy = String(isPending);
    row.disabled =
      quarantined ||
      state.deletingThreadTokens.has(state.activeThreadId) ||
      state.clearingThreadTokens.has(state.activeThreadId) ||
      state.uploadThreadTokens.has(state.activeThreadId) ||
      state.runningQuery?.threadId === state.activeThreadId;
    if (!quarantined) {
      row.addEventListener("click", () => inspectDurableRun(run.run_id));
    }

    const title = document.createElement("strong");
    title.textContent = quarantined
      ? "Quarantined · Stored loop run"
      : `${run.final_decision || "unknown"} · ${
          run.recipe_name || "Loop recipe"
        }`;

    const meta = document.createElement("span");
    if (quarantined) {
      const reasonLabel = LOOP_RUN_QUARANTINE_REASON_LABELS.get(
        run.quarantine_reason,
      ) || LOOP_RUN_QUARANTINE_REASON_LABELS.get(DEFAULT_QUARANTINE_REASON);
      meta.textContent = `${reasonLabel} · ${run.run_id}`;
    } else {
      const stepCount = Number.isSafeInteger(run.step_count) ? run.step_count : 0;
      meta.textContent = `${stepCount} ${stepCount === 1 ? "step" : "steps"} · ${
        run.context_provider || "none"
      } · ${run.backend || "backend"} · ${run.run_id}`;
    }

    row.append(title, meta);
    elements.runList.append(row);
  }
  renderRunInspectorStatus(durableRuns);
  restoreInspectedRunFocus();
}

function renderModelThinking(thinking) {
  const data = thinking || {};
  const hasThinking = Boolean(data.available && data.content);
  const isRedacted = Boolean(data.redacted);
  const label = data.label || "Model Thinking (unverified)";

  elements.thinkingPanel.querySelector("summary span").textContent = label;
  elements.thinkingNote.textContent =
    data.note ||
    "Model-emitted thinking is useful for debugging the loop, but it is not verified evidence.";
  elements.thinkingState.textContent = isRedacted
    ? "redacted"
    : hasThinking
      ? "captured"
      : "not captured";
  elements.thinkingState.dataset.state = isRedacted
    ? "redacted"
    : hasThinking
      ? "captured"
      : "empty";
  elements.thinkingContent.textContent = data.content
    ? data.content
    : "No model thinking captured for this run.";
  elements.thinkingPanel.open = hasThinking || isRedacted;
}

function renderLoopPayload(payload) {
  state.latest = payload;
  renderMemoryStatus(payload);
  renderActiveThreadTitle();
  renderTimeline(payload.timeline);
  renderModelThinking(payload.trace?.model_thinking);
  elements.summaryJson.textContent = JSON.stringify(payload.summary, null, 2);
  elements.traceJson.textContent = JSON.stringify(payload.trace, null, 2);
}

async function loadRecipes() {
  const payload = await requestJson("/api/recipes");
  state.recipes = Array.isArray(payload?.recipes)
    ? payload.recipes.map(sanitizeRecipe).filter((recipe) => recipe.recipe_id)
    : [];
  const savedRecipeId = loadActiveRecipeId();
  const defaultRecipeId = String(payload?.default_recipe_id || "");
  const preferredRecipeId =
    savedRecipeId && state.recipes.some((recipe) => recipe.recipe_id === savedRecipeId)
      ? savedRecipeId
      : defaultRecipeId;
  state.activeRecipeId = state.recipes.some(
    (recipe) => recipe.recipe_id === preferredRecipeId,
  )
    ? preferredRecipeId
    : state.recipes[0]?.recipe_id || "";
  persistActiveRecipeId();
  if (state.activeRecipeId) {
    await loadRecipeDetail(state.activeRecipeId).catch(() => null);
  }
  renderRecipes();
}

function activeRecipe() {
  return (
    state.recipes.find((recipe) => recipe.recipe_id === state.activeRecipeId) ||
    state.recipes[0] ||
    null
  );
}

function upsertRecipe(recipe) {
  if (!recipe.recipe_id) {
    return;
  }
  const others = state.recipes.filter((item) => item.recipe_id !== recipe.recipe_id);
  state.recipes = [recipe, ...others].sort((left, right) => {
    if (left.is_default !== right.is_default) {
      return left.is_default ? -1 : 1;
    }
    return String(left.name).localeCompare(String(right.name));
  });
}

async function loadRecipeDetail(recipeId) {
  const recipe = sanitizeRecipe(
    await requestJson(`/api/recipes/${encodeURIComponent(recipeId)}`),
  );
  upsertRecipe(recipe);
  return recipe;
}

function recipeFields() {
  return [
    elements.recipeName,
    elements.recipeGoal,
    elements.recipeInstructions,
    elements.recipeCriteria,
    elements.recipeStop,
  ];
}

function renderRecipeEditor() {
  const recipe = state.recipeDraft || activeRecipe();
  const isDraft = Boolean(state.recipeDraft);
  const isDefault = Boolean(recipe?.is_default);
  const detailLoaded = isDraft || isDefault || Boolean(recipe?.detail_loaded);
  elements.recipeEditorState.textContent = isDraft
    ? "new"
    : isDefault
      ? "default"
      : detailLoaded
        ? "custom"
        : "loading";
  elements.recipeName.value = recipe?.name || "";
  elements.recipeGoal.value = recipe?.goal || "";
  elements.recipeInstructions.value = recipe?.instructions || "";
  elements.recipeCriteria.value = Array.isArray(recipe?.success_criteria)
    ? recipe.success_criteria.join("\n")
    : "";
  elements.recipeStop.value = recipe?.stop_condition || "";
  for (const field of recipeFields()) {
    field.readOnly = !isDraft && (isDefault || !detailLoaded);
  }
  elements.recipeSaveButton.disabled =
    !recipe || (!isDraft && (isDefault || !detailLoaded));
  elements.recipeDeleteButton.disabled = !recipe || isDraft || isDefault;
  elements.recipeExportButton.disabled = !recipe || isDraft;
}

function renderRecipes() {
  elements.recipeSelect.replaceChildren();
  for (const recipe of state.recipes) {
    const option = document.createElement("option");
    option.value = recipe.recipe_id;
    option.textContent = recipe.name;
    elements.recipeSelect.append(option);
  }
  elements.recipeSelect.value = state.activeRecipeId || "";
  const recipe = activeRecipe();
  elements.recipeState.textContent = recipe?.is_default ? "default" : "custom";
  elements.recipeStatus.textContent = recipe
    ? safeText(recipe.goal || recipe.description || recipe.name, "", 180)
    : "No loop recipe available.";
  renderRecipeEditor();
}

function recipePayloadFromForm() {
  const basis = state.recipeDraft || activeRecipe() || {};
  return {
    name: elements.recipeName.value,
    goal: elements.recipeGoal.value,
    instructions: elements.recipeInstructions.value,
    success_criteria: elements.recipeCriteria.value
      .split(/\r?\n/)
      .map((criterion) => criterion.trim())
      .filter(Boolean),
    stop_condition: elements.recipeStop.value,
    context_provider: basis.context_provider || "smart",
    model_profile: basis.model_profile || "quality",
    verifier: basis.verifier || "default",
    metadata: basis.metadata || {},
  };
}

function startNewRecipe() {
  state.recipeDraft = {
    recipe_id: "",
    name: "",
    goal: "",
    instructions: "",
    success_criteria: [],
    stop_condition: "",
    context_provider: "smart",
    model_profile: "quality",
    verifier: "default",
    is_default: false,
  };
  elements.recipeEditor.open = true;
  elements.recipeStatus.textContent = "Editing new custom recipe.";
  renderRecipeEditor();
  elements.recipeName.focus();
}

async function saveRecipe() {
  let active = activeRecipe();
  const draft = state.recipeDraft;
  setBusy(elements.recipeSaveButton, true, "Save");
  try {
    if (!draft && active && !active.detail_loaded) {
      elements.recipeStatus.textContent = "Loading recipe details...";
      active = await loadRecipeDetail(active.recipe_id);
      renderRecipes();
    }
    const payload = recipePayloadFromForm();
    const saved = sanitizeRecipe(
      await requestJson(
        draft
          ? "/api/recipes"
          : `/api/recipes/${encodeURIComponent(active.recipe_id)}`,
        {
          method: draft ? "POST" : "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        },
      ),
    );
    state.recipeDraft = null;
    upsertRecipe(saved);
    state.activeRecipeId = saved.recipe_id;
    persistActiveRecipeId();
    renderRecipes();
    elements.recipeStatus.textContent = `Saved ${saved.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    setBusy(elements.recipeSaveButton, false, "Save");
    renderRecipeEditor();
  }
}

async function deleteRecipe() {
  const recipe = activeRecipe();
  if (!recipe || recipe.is_default) {
    return;
  }
  const confirmed =
    typeof globalThis.confirm === "function"
      ? globalThis.confirm(`Delete ${recipe.name}?`)
      : true;
  if (!confirmed) {
    return;
  }
  setBusy(elements.recipeDeleteButton, true, "Delete");
  try {
    await requestJson(`/api/recipes/${encodeURIComponent(recipe.recipe_id)}`, {
      method: "DELETE",
    });
    state.recipes = state.recipes.filter(
      (item) => item.recipe_id !== recipe.recipe_id,
    );
    state.activeRecipeId = state.recipes[0]?.recipe_id || "";
    persistActiveRecipeId();
    if (state.activeRecipeId) {
      await loadRecipeDetail(state.activeRecipeId).catch(() => null);
    }
    renderRecipes();
    elements.recipeStatus.textContent = `Deleted ${recipe.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    setBusy(elements.recipeDeleteButton, false, "Delete");
    renderRecipeEditor();
  }
}

function downloadRecipe(recipe) {
  const blob = new Blob([JSON.stringify(recipe, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${recipe.recipe_id || "loop-recipe"}.json`;
  document.body?.append(link);
  link.click();
  link.remove?.();
  URL.revokeObjectURL(url);
}

async function exportRecipe() {
  const recipe = activeRecipe();
  if (!recipe) {
    return;
  }
  try {
    const exported = await requestJson(
      `/api/recipes/${encodeURIComponent(recipe.recipe_id)}/export`,
    );
    downloadRecipe(exported);
    elements.recipeStatus.textContent = `Exported ${recipe.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  }
}

async function importRecipe() {
  const file = elements.recipeImport.files?.[0];
  if (!file) {
    return;
  }
  try {
    const raw = JSON.parse(await file.text());
    const imported = sanitizeRecipe(
      await requestJson("/api/recipes", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(raw),
      }),
    );
    state.recipeDraft = null;
    upsertRecipe(imported);
    state.activeRecipeId = imported.recipe_id;
    persistActiveRecipeId();
    renderRecipes();
    elements.recipeStatus.textContent = `Imported ${imported.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    elements.recipeImport.value = "";
  }
}

async function refreshStatus() {
  const requestThreadId = state.activeThreadId;
  const headers = {};
  if (requestThreadId) {
    headers["x-ai-loop-session-id"] = requestThreadId;
  }
  let status;
  try {
    status = await requestJson("/api/status", { headers });
  } catch (error) {
    if (state.activeThreadId === requestThreadId) {
      throw error;
    }
    return false;
  }
  if (state.activeThreadId !== requestThreadId) {
    return false;
  }
  renderRuntimeStatus(status);
  return true;
}

async function refreshStatusAfterStaleUpload() {
  if (await refreshStatus()) {
    elements.uploadStatus.textContent = "Active session changed; current file status refreshed.";
  }
}

async function uploadDocument(event) {
  event?.preventDefault?.();
  const file = elements.fileInput.files[0];
  if (!file) {
    elements.uploadStatus.textContent = "Choose a file first.";
    return;
  }

  const uploadThreadId = activeThread().id || "default";
  if (!uploadThreadId || threadMutationInProgress(uploadThreadId)) {
    return;
  }
  const operationToken = state.threadOperationRequestId + 1;
  state.threadOperationRequestId = operationToken;
  state.uploadThreadTokens.set(uploadThreadId, operationToken);
  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_encoding", elements.textEncoding?.value || DEFAULT_TEXT_ENCODING);
  formData.append("session_id", uploadThreadId);

  setBusy(elements.uploadButton, true, "Attach File");
  setQueryControlsBusy(true);
  renderThreads();
  elements.uploadStatus.textContent = "Attaching file to this session...";
  try {
    const result = await requestJson("/api/documents", {
      method: "POST",
      body: formData,
    });
    if (uploadStillMatchesActiveThread(uploadThreadId, operationToken)) {
      renderRuntimeStatus(result.status, {
        uploadMessage: sessionFileMessage(result.status),
      });
    } else {
      await refreshStatusAfterStaleUpload().catch((refreshError) => {
        elements.uploadStatus.textContent = refreshError.message;
      });
    }
  } catch (error) {
    if (uploadStillMatchesActiveThread(uploadThreadId, operationToken)) {
      elements.uploadStatus.textContent = error.message;
      await refreshStatus().catch(() => {});
    } else {
      await refreshStatusAfterStaleUpload().catch((refreshError) => {
        elements.uploadStatus.textContent = refreshError.message;
      });
    }
  } finally {
    if (
      ownsThreadOperation(
        state.uploadThreadTokens,
        uploadThreadId,
        operationToken,
      )
    ) {
      state.uploadThreadTokens.delete(uploadThreadId);
    }
    if (!state.uploadThreadTokens.size) {
      setBusy(elements.uploadButton, false, "Attach File");
    }
    if (!state.runningQuery && !threadMutationInProgress(state.activeThreadId)) {
      setQueryControlsBusy(false);
    }
    renderThreads();
  }
}

function chooseDocumentFile() {
  if (elements.uploadButton.disabled) {
    return;
  }
  elements.fileInput.click();
}

async function runQuery(event) {
  event.preventDefault();
  const message = elements.queryInput.value.trim();
  if (!message) {
    return;
  }

  const requestThread = activeThread();
  const requestThreadId = requestThread.id;
  if (
    state.runningQuery ||
    !requestThreadId ||
    threadMutationInProgress(requestThreadId)
  ) {
    return;
  }
  const baseMessageCount = nonNegativeInteger(requestThread.messageCount);
  const queryId = `query_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const requestRevision = bumpThreadRevision(requestThread);
  const isFirstUserMessage = !requestThread.messages.some(
    (item) => item.role === "user",
  );
  requestThread.messages.push({
    role: "user",
    content: message,
    local_query_id: queryId,
  });
  if (isFirstUserMessage) {
    requestThread.title = titleFromMessage(message);
  }
  const pendingMessage = pendingAssistantMessage({
    contextLabel: selectedEvidenceLabel(),
    recipeName: activeRecipeName(),
    queryId,
    userContent: message,
    baseMessageCount,
  });
  requestThread.messages.push(pendingMessage);
  touchThread(requestThread);
  setQueryControlsBusy(true);
  startQueryProgress(requestThreadId, pendingMessage.pending_id);
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  resetRunInspection("Showing live run progress.");
  renderRuns(requestThread.loopRuns);
  renderLoopPayload(runningLoopPayload(pendingMessage));
  elements.queryInput.value = "";
  try {
    const queryPayload = {
      message,
      session_id: requestThreadId,
      recipe_id: state.activeRecipeId || undefined,
    };
    const contextProvider = String(elements.queryContext?.value || "").trim();
    if (contextProvider) {
      queryPayload.context_provider = contextProvider;
    }
    const result = await requestJson("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(queryPayload),
    });
    const targetThread = threadById(requestThreadId);
    if (
      !targetThread ||
      normalizedRevision(targetThread.revision) !== requestRevision ||
      state.deletingThreadTokens.has(requestThreadId) ||
      state.clearingThreadTokens.has(requestThreadId)
    ) {
      return;
    }
    const removedPending = removePendingMessage(
      targetThread,
      pendingMessage.pending_id,
    );
    if (removedPending) {
      targetThread.messages.push({
        role: "assistant",
        content: result.answer,
        thinking: result.trace?.model_thinking || null,
      });
    }
    if (result.run?.run_id) {
      const run = sanitizeLoopRun(result.run, requestThreadId);
      if (!run.run_id) {
        throw new Error("Stored run summary is unavailable or malformed.");
      }
      const priorRunCount = Number.isSafeInteger(targetThread.loopRunCount)
        ? targetThread.loopRunCount
        : Array.isArray(targetThread.loopRuns) ? targetThread.loopRuns.length : 0;
      const hadRun = Array.isArray(targetThread.loopRuns)
        ? targetThread.loopRuns.some((item) => item.run_id === run.run_id)
        : false;
      targetThread.loopRuns = [
        run,
        ...(targetThread.loopRuns || []).filter((item) => item.run_id !== run.run_id),
      ];
      targetThread.loopRunCount = Math.max(
        hadRun ? priorRunCount : priorRunCount + 1,
        targetThread.loopRuns.length,
      );
    }
    if (result.thread) {
      applyThreadSummary(targetThread, sanitizeThread(result.thread));
    }
    targetThread.latest = result;
    touchThread(targetThread);
    renderThreads();
    if (state.activeThreadId === requestThreadId) {
      resetRunInspection();
      renderActiveThreadTitle();
      renderMessages();
      renderRuns(targetThread.loopRuns);
      renderLoopPayload(result);
    }
  } catch (error) {
    const targetThread = threadById(requestThreadId);
    if (
      !targetThread ||
      normalizedRevision(targetThread.revision) !== requestRevision ||
      state.deletingThreadTokens.has(requestThreadId) ||
      state.clearingThreadTokens.has(requestThreadId)
    ) {
      return;
    }
    removePendingMessage(targetThread, pendingMessage.pending_id);
    targetThread.messages.push({ role: "assistant", content: error.message });
    touchThread(targetThread);
    renderThreads();
    if (state.activeThreadId === requestThreadId) {
      resetRunInspection("Showing the latest query error.");
      renderRuns(targetThread.loopRuns);
      renderMessages();
      renderLoopPayload(queryErrorLoopPayload(error.message));
    }
  } finally {
    const ownsRunningState =
      state.runningQuery?.pendingId === pendingMessage.pending_id;
    stopQueryProgress(pendingMessage.pending_id);
    if (
      state.activeThreadId === requestThreadId &&
      !state.deletingThreadTokens.has(requestThreadId) &&
      !state.clearingThreadTokens.has(requestThreadId) &&
      !state.uploadThreadTokens.has(requestThreadId)
    ) {
      renderRuns(activeThread().loopRuns);
    }
    renderThreads();
    if (
      (ownsRunningState || !state.runningQuery) &&
      !state.deletingThreadTokens.has(state.activeThreadId) &&
      !state.clearingThreadTokens.has(state.activeThreadId) &&
      !state.uploadThreadTokens.has(state.activeThreadId)
    ) {
      setQueryControlsBusy(false);
    }
  }
}

async function clearChat() {
  const thread = activeThread();
  const threadId = thread.id;
  if (!threadId || threadMutationInProgress(threadId)) {
    return;
  }
  const rollbackSnapshot = {
    ...thread,
    messages: thread.messages.map((message) => ({ ...message })),
    loopRuns: thread.loopRuns.map((run) => ({ ...run })),
  };
  const operationToken = state.threadOperationRequestId + 1;
  state.threadOperationRequestId = operationToken;
  state.clearingThreadTokens.set(threadId, operationToken);
  const clearRevision = bumpThreadRevision(thread);
  thread.messages = [];
  thread.loopRuns = [];
  thread.loopRunCount = 0;
  thread.memoryCount = 0;
  thread.latest = null;
  resetRunInspection();
  touchThread(thread);
  setQueryControlsBusy(true);
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(thread.loopRuns);
  renderLoopPayload(emptyLoopPayload());
  try {
    const payload = await requestJson("/api/chat/clear", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: threadId }),
    });
    const targetThread = threadById(threadId);
    if (
      !ownsThreadOperation(
        state.clearingThreadTokens,
        threadId,
        operationToken,
      ) ||
      !targetThread ||
      normalizedRevision(targetThread.revision) !== clearRevision
    ) {
      return;
    }
    targetThread.latest = payload;
    touchThread(targetThread);
    if (state.activeThreadId === threadId) {
      renderRuns(targetThread.loopRuns);
      renderLoopPayload(payload);
    }
  } catch (error) {
    let reconciled = false;
    let restoredFromServer = false;
    try {
      reconciled = await reconcileThreadDetail({
        threadId,
        expectedRevision: clearRevision,
        tokens: state.clearingThreadTokens,
        token: operationToken,
      });
      restoredFromServer = reconciled;
    } catch {
      const current = threadById(threadId);
      if (
        ownsThreadOperation(
          state.clearingThreadTokens,
          threadId,
          operationToken,
        ) &&
        current &&
        normalizedRevision(current.revision) === clearRevision
      ) {
        rollbackSnapshot.revision = clearRevision + 1;
        upsertThread(rollbackSnapshot);
        reconciled = true;
      }
    }
    if (
      reconciled &&
      ownsThreadOperation(
        state.clearingThreadTokens,
        threadId,
        operationToken,
      ) &&
      state.activeThreadId === threadId
    ) {
      resetRunInspection(
        restoredFromServer
          ? "Clear failed; refreshed the durable thread state."
          : "Clear status unavailable; restored the previous local view.",
      );
      renderActiveThreadTitle();
      renderMessages();
      renderRuns(activeThread().loopRuns);
      renderLoopPayload(activeThread().latest || emptyLoopPayload());
    }
    if (
      ownsThreadOperation(
        state.clearingThreadTokens,
        threadId,
        operationToken,
      ) &&
      state.activeThreadId === threadId
    ) {
      elements.uploadStatus.textContent = error.message;
    }
  } finally {
    if (
      ownsThreadOperation(
        state.clearingThreadTokens,
        threadId,
        operationToken,
      )
    ) {
      state.clearingThreadTokens.delete(threadId);
    }
    if (
      !state.runningQuery &&
      !state.deletingThreadTokens.has(state.activeThreadId) &&
      !state.clearingThreadTokens.has(state.activeThreadId) &&
      !state.uploadThreadTokens.has(state.activeThreadId)
    ) {
      setQueryControlsBusy(false);
    }
    renderThreads();
  }
}

function setupTabs() {
  for (const button of elements.tabButtons) {
    button.addEventListener("click", () => {
      for (const tab of elements.tabButtons) {
        tab.classList.toggle("active", tab === button);
      }
      elements.summaryJson.classList.toggle(
        "hidden",
        button.dataset.target !== "summary-json",
      );
      elements.traceJson.classList.toggle(
        "hidden",
        button.dataset.target !== "trace-json",
      );
    });
  }
}

async function boot() {
  setupTabs();
  await loadRecipes();
  await loadThreads();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  resetRunInspection();
  renderRuns(activeThread().loopRuns);
  renderLoopPayload(activeThread().latest || emptyLoopPayload());
  elements.uploadButton.addEventListener("click", chooseDocumentFile);
  elements.fileInput.addEventListener("change", uploadDocument);
  elements.queryForm.addEventListener("submit", runQuery);
  elements.newThreadButton.addEventListener("click", () =>
    startNewThread().catch((error) => {
      elements.uploadStatus.textContent = error.message;
    }),
  );
  elements.deleteThreadButton.addEventListener("click", () =>
    deleteActiveThread().catch((error) => {
      elements.uploadStatus.textContent = error.message;
    }),
  );
  elements.clearButton.addEventListener("click", clearChat);
  if (elements.refreshStatus) {
    elements.refreshStatus.addEventListener("click", refreshStatus);
  }
  elements.recipeSelect.addEventListener("change", () => {
    state.activeRecipeId = elements.recipeSelect.value;
    state.recipeDraft = null;
    persistActiveRecipeId();
    loadRecipeDetail(state.activeRecipeId)
      .then(renderRecipes)
      .catch((error) => {
        elements.recipeStatus.textContent = error.message;
        renderRecipeEditor();
      });
  });
  elements.recipeNewButton.addEventListener("click", startNewRecipe);
  elements.recipeSaveButton.addEventListener("click", saveRecipe);
  elements.recipeDeleteButton.addEventListener("click", deleteRecipe);
  elements.recipeExportButton.addEventListener("click", exportRecipe);
  elements.recipeImport.addEventListener("change", importRecipe);
  await refreshStatus();
}

boot().catch((error) => {
  elements.uploadStatus.textContent = error.message;
});
