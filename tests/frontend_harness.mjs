import { readFile } from "node:fs/promises";

class ClassList {
  constructor(element) {
    this.element = element;
  }

  toggle(name, force) {
    const tokens = new Set(this.element.className.split(/\s+/).filter(Boolean));
    const enabled = force === undefined ? !tokens.has(name) : Boolean(force);
    if (enabled) {
      tokens.add(name);
    } else {
      tokens.delete(name);
    }
    this.element.className = Array.from(tokens).join(" ");
  }
}

export class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.listeners = {};
    this.open = false;
    this.readOnly = false;
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.textContent = "";
    this.value = "";
    this.classList = new ClassList(this);
  }

  append(...nodes) {
    for (const node of nodes) {
      node.parentElement = this;
      this.children.push(node);
    }
    this.scrollHeight = this.children.length;
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }

  remove() {
    if (!this.parentElement) {
      return;
    }
    this.parentElement.children = this.parentElement.children.filter(
      (child) => child !== this,
    );
    this.parentElement = null;
  }

  addEventListener(type, listener) {
    this.listeners[type] = listener;
  }

  async dispatch(type) {
    const listener = this.listeners[type];
    if (!listener) {
      return;
    }
    await listener({ preventDefault() {} });
  }

  async click() {
    if (this.listeners.click) {
      await this.dispatch("click");
      return;
    }
    if (this.tagName === "A") {
      globalThis.__downloads.push({
        href: this.href || "",
        download: this.download || "",
      });
    }
  }

  focus() {}

  querySelector(selector) {
    if (selector === "summary span") {
      const summary = findNode(this, (node) => node.tagName === "SUMMARY");
      return summary ? findNode(summary, (node) => node.tagName === "SPAN") : null;
    }
    return findNode(this, (node) => matches(node, selector));
  }
}

function matches(node, selector) {
  if (selector.startsWith(".")) {
    return node.className.split(/\s+/).includes(selector.slice(1));
  }
  return node.tagName === selector.toUpperCase();
}

export function findNode(root, predicate) {
  for (const child of root.children) {
    if (predicate(child)) {
      return child;
    }
    const nested = findNode(child, predicate);
    if (nested) {
      return nested;
    }
  }
  return null;
}

export function nodeText(root) {
  return [
    root.textContent || "",
    ...root.children.map((child) => nodeText(child)),
  ].join(" ");
}

function createMemoryStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

export function createDom() {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: createMemoryStorage(),
  });
  Object.defineProperty(globalThis, "__downloads", {
    configurable: true,
    value: [],
  });
  Object.defineProperty(globalThis, "__objectUrls", {
    configurable: true,
    value: [],
  });
  Object.defineProperty(globalThis, "__revokedObjectUrls", {
    configurable: true,
    value: [],
  });
  globalThis.URL.createObjectURL = (blob) => {
    const url = `blob:test-${globalThis.__objectUrls.length + 1}`;
    globalThis.__objectUrls.push({ url, blob });
    return url;
  };
  globalThis.URL.revokeObjectURL = (url) => {
    globalThis.__revokedObjectUrls.push(url);
  };
  const ids = [
    "backend-pill",
    "model-pill",
    "ready-pill",
    "upload-button",
    "upload-status",
    "document-file",
    "text-encoding",
    "new-thread",
    "delete-thread",
    "thread-list",
    "recipe-select",
    "recipe-status",
    "recipe-state",
    "recipe-editor",
    "recipe-editor-state",
    "recipe-new",
    "recipe-save",
    "recipe-delete",
    "recipe-export",
    "recipe-import",
    "recipe-name",
    "recipe-goal",
    "recipe-instructions",
    "recipe-criteria",
    "recipe-stop",
    "active-thread-title",
    "active-thread-memory",
    "file-scope",
    "query-form",
    "query-context",
    "query-input",
    "query-button",
    "clear-chat",
    "messages",
    "memory-status",
    "timeline",
    "run-list",
    "run-count",
    "final-decision",
    "thinking-panel",
    "thinking-state",
    "thinking-note",
    "thinking-content",
    "summary-json",
    "trace-json",
  ];
  const byId = Object.fromEntries(ids.map((id) => [id, new Element("div")]));
  byId["query-context"].value = "";
  byId["query-input"].value = "";
  const summary = new Element("summary");
  summary.append(new Element("span"));
  byId["thinking-panel"].append(summary);
  const tabButtons = [new Element("button"), new Element("button")];
  tabButtons[0].dataset.target = "summary-json";
  tabButtons[1].dataset.target = "trace-json";

  globalThis.document = {
    body: new Element("body"),
    createElement(tagName) {
      return new Element(tagName);
    },
    querySelector(selector) {
      if (!selector.startsWith("#")) {
        return null;
      }
      return byId[selector.slice(1)] || null;
    },
    querySelectorAll(selector) {
      return selector === ".tab-button" ? tabButtons : [];
    },
  };
  return byId;
}

export function jsonResponse(payload) {
  return {
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    async json() {
      return payload;
    },
    async text() {
      return JSON.stringify(payload);
    },
  };
}

export function errorResponse(status, payload) {
  return {
    ok: false,
    status,
    headers: { get: () => "application/json" },
    async json() {
      return payload;
    },
    async text() {
      return JSON.stringify(payload);
    },
  };
}

export function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

export function createThreadPayload(id, messages = [], latest = null, options = {}) {
  const payload = {
    id,
    title: "New thread",
    messages,
    latest,
    message_count: messages.length,
    memory_count: options.memoryCount || 0,
    loop_run_count: options.loopRunCount || 0,
    created_at: "2026-06-26T00:00:00.000Z",
    updated_at: "2026-06-26T00:00:00.000Z",
  };
  if (options.includeRuns !== false) {
    payload.loop_runs = options.loopRuns || [];
  }
  return payload;
}

export async function importFreshApp() {
  const source = await readFile(process.env.APP_JS_PATH, "utf8");
  const encoded = Buffer.from(
    `${source}\n// frontend harness case ${Date.now()} ${Math.random()}`,
    "utf8",
  ).toString("base64");
  await import(`data:text/javascript;base64,${encoded}`);
}

export async function tick() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}
