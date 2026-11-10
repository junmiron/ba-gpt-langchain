/* Business Analyst agent — single-page UI (vanilla JS, no build step).
 *
 * Data flow: REST for commands, SSE for everything the agent produces. The timeline is rebuilt from stored
 * events when a session is opened, then kept live by the event stream; token/reasoning frames are live-only.
 */

"use strict";

const PHASES = [
  { id: "intake", label: "Plan" },
  { id: "interview", label: "Interview" },
  { id: "clarify", label: "Clarify" },
  { id: "summary", label: "Summary" },
  { id: "approval", label: "Approval" },
  { id: "draft", label: "Draft" },
  { id: "review", label: "Review" },
  { id: "diagram", label: "Diagram" },
  { id: "deliver", label: "Deliver" },
  { id: "done", label: "Done" },
];

const ROLE_LABELS = {
  plan: "Planner", interview: "Interviewer", triage: "Analyst", summary: "Summariser", writer: "Spec writer",
  use_cases: "Use-case analyst", glossary: "Glossarist", critic: "Reviewer", diagrammer: "Diagrammer",
  mermaid_reviewer: "Mermaid reviewer", human: "Human", chat: "BA assistant", engine: "Workflow",
};

let activityStartedAt = 0;
let activityPhase = "";
let activityTimer = null;

const S = {
  sessionId: null,
  sessionOffset: 0,
  sessionTotal: 0,
  snapshot: null,
  models: [],
  defaultModel: null,
  overrides: {},
  embeddingModel: null,
  es: null,
  lastEventId: 0,
  renderedIds: new Set(),
  openStreams: new Map(),   // key → element
  diagrams: [],
  reconnectTimer: null,
  settings: null,
  transcriptSource: "",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ utilities */

// Bearer token for the UI when `server.auth_token` is set. Stored in localStorage so a refresh does not
// ask again; the static page itself is always served, only /api/* is gated.
const TOKEN_KEY = "ba-agent-token";

function authToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

async function api(path, options = {}) {
  const init = { headers: {}, ...options };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const token = authToken();
  if (token) init.headers["Authorization"] = `Bearer ${token}`;
  const response = await fetch(path, init);
  if (response.status === 401) {
    // The server wants a token we do not have yet: ask once, then retry.
    const supplied = window.prompt("API token (server.auth_token):", "");
    if (supplied) {
      localStorage.setItem(TOKEN_KEY, supplied.trim());
      return api(path, options);
    }
    throw new Error("the API is token-protected and no token was supplied");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* keep default */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " err" : "");
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 3200);
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const ALLOWED_TAGS = new Set(["P", "BR", "HR", "H1", "H2", "H3", "H4", "H5", "H6", "UL", "OL", "LI", "STRONG",
  "EM", "CODE", "PRE", "BLOCKQUOTE", "TABLE", "THEAD", "TBODY", "TR", "TH", "TD", "A", "IMG", "DEL", "SPAN", "SUP", "SUB"]);

/** Render markdown with marked.js, then strip anything that could execute. */
function safeHTML(markdown) {
  const raw = window.marked ? window.marked.parse(String(markdown ?? ""), { gfm: true, breaks: false }) : `<pre>${escapeHtml(markdown)}</pre>`;
  const doc = new DOMParser().parseFromString(raw, "text/html");
  const walk = (node) => {
    Array.from(node.children).forEach((child) => {
      if (!ALLOWED_TAGS.has(child.tagName)) {
        child.replaceWith(...Array.from(child.childNodes));
        return;
      }
      Array.from(child.attributes).forEach((attr) => {
        const name = attr.name.toLowerCase();
        const dangerous = name.startsWith("on") || ((name === "href" || name === "src") && /^\s*javascript:/i.test(attr.value));
        if (dangerous) child.removeAttribute(attr.name);
      });
      if (child.tagName === "A") { child.target = "_blank"; child.rel = "noopener noreferrer"; }
      walk(child);
    });
  };
  doc.body.setAttribute("class", "markdown-body");
  walk(doc.body);
  return doc.body.innerHTML;
}

function fmtTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return isNaN(date) ? "" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function bytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

/* ------------------------------------------------------------------- timeline */

function card({ role = "engine", title = "", kind = "", time = "", tags = [] }) {
  const el = document.createElement("article");
  el.className = `card ${kind}`;
  const header = document.createElement("header");
  const label = ROLE_LABELS[role] || (role ? role.replace(/_/g, " ") : "Workflow");
  header.innerHTML = `<span class="role-badge">${escapeHtml(label)}</span>` +
    tags.map((t) => `<span class="tag ${t.cls || ""}">${escapeHtml(t.text)}</span>`).join("") +
    (title ? `<span>${escapeHtml(title)}</span>` : "") +
    `<span class="time">${fmtTime(time)}</span>`;
  const body = document.createElement("div");
  body.className = "body";
  el.append(header, body);
  return { el, body, header };
}

function timelineIsNearBottom(timeline, threshold = 160) {
  return timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < threshold;
}

function pushCard(el) {
  const timeline = $("#timeline");
  const followLatest = timelineIsNearBottom(timeline);
  const welcome = timeline.querySelector(".welcome");
  if (welcome) welcome.remove();
  timeline.appendChild(el);
  if (followLatest) timeline.scrollTop = timeline.scrollHeight;
}

function closeStreams() { S.openStreams.clear(); }

function appendToken(role, step, text) {
  const timeline = $("#timeline");
  const followLatest = timelineIsNearBottom(timeline);
  const key = `${role}:${step}`;
  let entry = S.openStreams.get(key);
  if (!entry) {
    const { el, body } = card({ role, kind: "streaming" });
    body.innerHTML = '<span class="streaming-caret"></span>';
    entry = { el, body, text: "" };
    S.openStreams.set(key, entry);
    pushCard(el);
  }
  entry.text += text;
  entry.body.innerHTML = safeHTML(entry.text) + '<span class="streaming-caret"></span>';
  if (followLatest) timeline.scrollTop = timeline.scrollHeight;
}

function finalizeStream(role, step) {
  const key = `${role}:${step}`;
  const entry = S.openStreams.get(key);
  if (!entry) return;
  entry.body.innerHTML = safeHTML(entry.text);
  S.openStreams.delete(key);
}

let refreshTimer = null;

/** Coalesce the many events of a step into one snapshot refresh (header counts, ledger, files). */
function scheduleRefresh(delay = 250) {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { refreshSnapshot().catch(() => { /* stream still authoritative */ }); }, delay);
}

let sessionListTimer = null;
/** The sidebar shows each session's phase and busy dot, so it has to move when a run moves. */
function scheduleSessionListRefresh(delay = 1500) {
  clearTimeout(sessionListTimer);
  sessionListTimer = setTimeout(() => { loadSessions().catch(() => {}); }, delay);
}

function traceLine(event) {
  const pane = $("#tab-trace");
  const line = document.createElement("div");
  line.className = "trace-line";
  const payload = event.payload || {};
  const extra = payload.category === "model_call"
    ? `${payload.role || "model"} · ${payload.step || ""} · ${payload.latency_ms || 0} ms · ${payload.method || ""} · ${payload.prompt_chars || 0} prompt chars`
    : (payload.phase || payload.section_id || payload.name || payload.tool || payload.model || "");
  line.innerHTML = `<b>${escapeHtml(event.type)}</b> ${escapeHtml(extra)} <span class="muted">${fmtTime(event.ts)}</span>`;
  pane.appendChild(line);
  pane.parentElement.scrollTop = pane.parentElement.scrollHeight;
}

/* ------------------------------------------------------------------ rendering */

function renderStepper(snapshot) {
  const stepper = $("#stepper");
  stepper.innerHTML = "";
  if (!snapshot) return;
  const currentIndex = PHASES.findIndex((p) => p.id === snapshot.phase);
  PHASES.forEach((phase, index) => {
    const step = document.createElement("span");
    let cls = "step";
    if (snapshot.status === "error" && index === currentIndex) cls += " error";
    else if (index < currentIndex || snapshot.phase === "done") cls += " done";
    else if (index === currentIndex) cls += snapshot.awaiting !== "none" ? " waiting" : " active";
    step.className = cls;
    step.innerHTML = `<i class="bullet"></i>${escapeHtml(phase.label)}`;
    stepper.appendChild(step);
  });
}

function startActivityIndicator(phase) {
  const box = $("#activity-indicator");
  if (!box) return;
  if (!activityStartedAt) activityStartedAt = Date.now();
  activityPhase = phase || activityPhase || "workflow";
  const labels = Object.fromEntries(PHASES.map((item) => [item.id, item.label]));
  const update = () => {
    const seconds = Math.floor((Date.now() - activityStartedAt) / 1000);
    const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    box.textContent = `Thinking · ${labels[activityPhase] || activityPhase} · ${elapsed}`;
  };
  box.hidden = false;
  update();
  if (!activityTimer) activityTimer = setInterval(update, 1000);
}

function stopActivityIndicator() {
  const box = $("#activity-indicator");
  if (activityTimer) clearInterval(activityTimer);
  activityTimer = null;
  activityStartedAt = 0;
  activityPhase = "";
  if (box) { box.hidden = true; box.textContent = ""; }
}

function renderHeader(snapshot) {
  if (!snapshot) return;
  $("#session-title").textContent = snapshot.project_name || snapshot.title || "Untitled";
  const bits = [];
  if (snapshot.objective) bits.push(snapshot.objective);
  bits.push(`${snapshot.questions_total} questions · ${snapshot.requirements_total} requirements`);
  if (snapshot.has_transcript) bits.push(`seeded from ${snapshot.transcript_source || "a transcript"}`);
  if (snapshot.metrics && snapshot.metrics.model_seconds) {
    bits.push(`model time ${Math.round(snapshot.metrics.model_seconds)}s / ${Math.round(snapshot.metrics.calls || 0)} calls`);
  }
  $("#session-meta").textContent = bits.join(" · ");
  const waiting = snapshot.awaiting !== "none" && snapshot.phase !== "done";
  $("#composer").hidden = !snapshot;
  if (snapshot.busy && snapshot.status === "running" && snapshot.awaiting === "none") startActivityIndicator(snapshot.phase);
  else stopActivityIndicator();
  $("#btn-stop").hidden = !snapshot.busy;
  // An errored run is resumable by design (e.g. LM Studio was off, or a model finished loading since).
  const resumable = ["idle", "cancelled", "error"].includes(snapshot.status) && snapshot.phase !== "done";
  $("#btn-resume").hidden = !resumable;
  return waiting;
}

function renderQuestionCard(snapshot) {
  const box = $("#question-card");
  const answerForm = $("#answer-form");
  const question = snapshot.pending_question;
  const showQuestion = snapshot.awaiting === "answer" && Boolean(question);
  answerForm.hidden = !showQuestion;
  box.hidden = !showQuestion;
  if (!showQuestion) { box.innerHTML = ""; renderApprovalCard(snapshot); return; }
  box.innerHTML = `
    <div class="q-subject">${escapeHtml(question.subject)} · ${escapeHtml(question.purpose)}</div>
    <div class="q-text">${escapeHtml(question.text)}</div>
    ${question.rationale ? `<div class="q-why">Why: ${escapeHtml(question.rationale)}</div>` : ""}
    ${question.options && question.options.length
      ? `<div class="q-options">${question.options.map((o) => `<button type="button" class="chip">${escapeHtml(o)}</button>`).join("")}</div>`
      : ""}`;
  box.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    const input = $("#answer-input");
    input.value = (input.value ? input.value + " " : "") + chip.textContent.trim();
    input.focus();
  }));
  renderApprovalCard(snapshot);
  $("#answer-input").focus();
}

function renderApprovalCard(snapshot) {
  const box = $("#approval-card");
  const awaitingApproval = snapshot.awaiting === "approval";
  box.hidden = !awaitingApproval;
  if (!awaitingApproval) { box.innerHTML = ""; return; }
  box.innerHTML = `
    <div class="q-subject">Summary approval required</div>
    <div class="q-text">Review the requirements summary in the Spec panel, then choose what to do next.</div>
    <div class="approval-actions">
      <button type="button" class="btn btn-primary btn-sm" data-approve="yes">Approve &amp; draft the FSD</button>
      <button type="button" class="btn btn-ghost btn-sm" data-approve="no">Request changes…</button>
    </div>`;
  box.querySelector('[data-approve="yes"]').addEventListener("click", () => approve(true, ""));
  box.querySelector('[data-approve="no"]').addEventListener("click", () => $("#dlg-reject").showModal());
}

function renderSpec(snapshot) {
  const pane = $("#tab-spec");
  if (!snapshot) return;
  const markdown = snapshot.spec_markdown || "";
  if (markdown.trim()) {
    pane.innerHTML = safeHTML(markdown);
  } else if (snapshot.summary_markdown) {
    const summaryStatus = snapshot.awaiting === "approval"
      ? "Approval required — review the summary below."
      : "Draft not started yet — summary below.";
    pane.innerHTML = `<p class="muted">${summaryStatus}</p>` + safeHTML(snapshot.summary_markdown);
  }
}

function renderLedger(snapshot) {
  const pane = $("#tab-ledger");
  const ledger = snapshot.ledger || [];
  if (!ledger.length) { pane.innerHTML = '<p class="muted">Nothing captured yet.</p>'; return; }
  const requirements = ledger.filter((i) => i.kind === "requirement");
  const others = ledger.filter((i) => i.kind !== "requirement");

  const reqRows = requirements.map((r) => `<tr>
      <td><b>${escapeHtml(r.id)}</b></td>
      <td>${escapeHtml(r.title || "")}</td>
      <td>${escapeHtml(r.category || "")}</td>
      <td>${escapeHtml(r.priority || "")}</td>
      <td>${r.acceptance && r.acceptance.length ? escapeHtml(r.acceptance.join("; ")) : '<span class="tag warn">missing</span>'}</td>
      <td class="muted">${escapeHtml(r.subject || "")}${r.source_q ? " · Q" + r.source_q : ""}</td>
      <td><button class="icon-btn" data-edit="${escapeHtml(r.id)}" title="Edit requirement">&#9998;</button></td>
    </tr>`).join("");

  const otherRows = others.map((i) => `<tr>
      <td><span class="tag">${escapeHtml(i.kind)}</span></td>
      <td>${escapeHtml(i.body || i.title || "")}</td>
      <td class="muted">${escapeHtml(i.subject || "")}</td>
      <td><button class="icon-btn" data-edit="${escapeHtml(i.id)}" title="Edit entry" aria-label="Edit ledger entry">&#9998;</button></td>
    </tr>`).join("");

  pane.innerHTML = `
    <p class="small muted">${requirements.length} requirement(s) · ${others.length} other ledger entries</p>
    ${requirements.length ? `<table class="data"><thead><tr><th>ID</th><th>Title</th><th>Category</th><th>Priority</th><th>Acceptance</th><th>Source</th></tr></thead><tbody>${reqRows}</tbody></table>` : ""}
    ${otherRows ? `<h3 style="font-size:13px;margin-top:18px">Facts, decisions, risks</h3><table class="data"><tbody>${otherRows}</tbody></table>` : ""}`;
  pane.querySelectorAll("[data-edit]").forEach((button) => {
    button.addEventListener("click", () => openLedgerEditor(button.dataset.edit));
  });
}

async function openLedgerEditor(itemId) {
  const entry = ((S.snapshot && S.snapshot.ledger) || []).find((i) => i.id === itemId);
  if (!entry) return;
  const dialog = $("#dlg-ledger");
  if (!dialog) return;
  dialog.dataset.itemId = itemId;
  dialog.dataset.kind = entry.kind || "";
  $("#ledger-title").value = entry.title || "";
  $("#ledger-body").value = entry.body || "";
  $("#ledger-priority").value = entry.priority || "must";
  $("#ledger-acceptance").value = (entry.acceptance || []).join("\n");
  const isRequirement = entry.kind === "requirement";
  $("#ledger-priority").hidden = !isRequirement;
  $("#ledger-acceptance").hidden = !isRequirement;
  $$("[data-field-label]").forEach((el) => { el.hidden = !isRequirement; });
  dialog.showModal();
}

function renderArtifacts(snapshot) {
  const pane = $("#tab-artifacts");
  const items = (snapshot && snapshot.artifacts) || [];
  if (!items.length) { pane.innerHTML = '<p class="muted">No artefacts yet.</p>'; return; }
  pane.innerHTML = `<div class="artifact-actions">
      <a class="btn btn-primary btn-sm" href="/api/sessions/${encodeURIComponent(S.sessionId)}/export" download>Download all (.zip)</a>
    </div>
    ${runFacts(snapshot.run_facts || {})}
    <ul class="artifact-list">${items.map((a) => `
    <li><a href="/api/artifacts/${encodeURIComponent(S.sessionId)}/${encodeURIComponent(a.name)}" download>
      <span>${escapeHtml(a.name)}</span><span class="muted small">${bytes(a.bytes)}</span></a></li>`).join("")}</ul>
    <p class="muted small">Files are also on disk under <code>outputs/${escapeHtml(S.sessionId)}/</code>.</p>`;
}

function runFacts(facts) {
  // Every value here comes from a JSON file we wrote ourselves, and is still escaped before it reaches
  // the DOM: a hand-edited metadata.json is a possible injection path.
  if (!facts || !Object.keys(facts).length) return "";
  const rows = [];
  if (facts.pdf_engine) {
    rows.push(`PDF engine: <b>${escapeHtml(facts.pdf_engine)}</b>${facts.pdf_available ? "" : " (not produced)"}`);
  }
  const rungs = (facts.structured_output || {}).rung_counts || {};
  if (Object.keys(rungs).length) {
    rows.push("Structured output: " + Object.entries(rungs).map(([rung, count]) => `${escapeHtml(rung)} × ${count}`).join(", "));
  }
  const usage = facts.llm_usage || {};
  if (usage.calls) {
    rows.push(`Model calls: <b>${usage.calls}</b>${usage.failures ? ` (${usage.failures} failed, retried)` : ""}`);
  }
  const phases = facts.phase_metrics || {};
  const slowest = Object.entries(phases)
    .filter(([, v]) => Number(v.wall_seconds) > 0)
    .sort((a, b) => Number(b[1].wall_seconds) - Number(a[1].wall_seconds))
    .slice(0, 3);
  if (slowest.length) {
    rows.push(
      "Slowest phases: " +
        slowest.map(([phase, v]) => `${escapeHtml(phase)} ${Math.round(v.wall_seconds)}s (${v.calls} calls)`).join(", ")
    );
  }
  if (facts.pages) rows.push(`PDF pages: ${facts.pages}`);
  if (facts.qa_ok !== undefined) rows.push(facts.qa_ok ? "QA gates: passed" : "QA gates: warnings");

  const warnings = (facts.qa_warnings || []).map((w) => `<li class="muted small">${escapeHtml(w)}</li>`);
  return `<div class="run-facts"><p class="muted small">${rows.join(" · ")}</p>${warnings.length ? `<ul>${warnings.join("")}</ul>` : ""}</div>`;
}

let mermaidSeq = 0;
async function renderDiagrams(diagrams, statusText = "") {
  const view = $("#diagram-view");
  const source = $("#diagram-source");
  $("#diagram-status").textContent = statusText;
  if (!diagrams || !diagrams.length) { view.innerHTML = '<p class="muted">No diagram yet.</p>'; source.hidden = true; return; }

  view.innerHTML = "";
  const sources = [];
  for (const diagram of diagrams) {
    const block = document.createElement("div");
    block.innerHTML = `<h3 style="margin:.4em 0;color:#1c2430;font-size:14px">${escapeHtml(diagram.title || "Diagram")}</h3>`;
    const holder = document.createElement("div");
    block.appendChild(holder);
    view.appendChild(block);
    sources.push(`### ${diagram.title || "Diagram"}\n\n${diagram.code}`);

    if (diagram.svg) { holder.innerHTML = diagram.svg; continue; }
    try {
      mermaidSeq += 1;
      const { svg } = await window.mermaid.render(`mmd-${mermaidSeq}`, diagram.code);
      holder.innerHTML = svg;
    } catch (error) {
      holder.innerHTML = `<p style="color:#b00020">Could not render: ${escapeHtml(error.message || error)}</p>` +
        `<pre class="code-block">${escapeHtml(diagram.code)}</pre>`;
    }
  }
  source.textContent = sources.join("\n\n");
  source.hidden = false;
}

/* ---------------------------------------------------------------- event stream */

// EventSource only delivers a frame to a listener registered for that exact `event:` name. A type
// missing from this list is dropped in silence: the run keeps emitting, the page shows nothing.
const STREAM_EVENT_TYPES = [
  "session", "step_start", "step_end", "token", "reasoning", "status", "question", "answer",
  "ledger", "spec", "diagram", "review", "tool_call", "artifact", "approval", "warning", "error", "log",
];

function connectStream(sessionId) {
  disconnectStream();
  // EventSource cannot set headers, so the token travels as a query parameter when auth is enabled.
  const token = authToken();
  const url = `/api/sessions/${sessionId}/stream?after=${S.lastEventId}${token ? `&token=${encodeURIComponent(token)}` : ""}`;
  const es = new EventSource(url);
  S.es = es;
  for (const type of STREAM_EVENT_TYPES) {
    es.addEventListener(type, (frame) => {
      handleEvent(JSON.parse(frame.data)).catch((error) => toast(`could not render ${type}: ${error.message}`, true));
    });
  }
  es.onopen = () => { $("#health-dot").className = "dot dot-on"; };
  es.onerror = () => {
    // Browser reconnects automatically and sends Last-Event-ID; we only surface the state.
    $("#health-dot").className = "dot dot-warn";
  };
}

function disconnectStream() {
  if (S.es) { S.es.close(); S.es = null; }
}

async function handleEvent(event) {
  const id = event.id || 0;
  if (id > 0) {
    if (S.renderedIds.has(id)) return;   // replay after reconnect — already on screen
    S.renderedIds.add(id);
    S.lastEventId = Math.max(S.lastEventId, id);
  }

  const payload = event.payload || {};
  const type = event.type;
  traceLine(event);
  if (type !== "token" && type !== "reasoning") scheduleRefresh();
  if (type === "step_end" || type === "question" || type === "approval" || type === "error") {
    scheduleSessionListRefresh();
  }

  switch (type) {
    case "token":
      appendToken(payload.role || "writer", payload.step || "", payload.text || "");
      return;

    case "reasoning":
      appendToken(payload.role || "writer", (payload.step || "") + ":thinking", payload.text || "");
      return;

    case "step_start":
      closeStreams();
      startActivityIndicator(payload.phase);
      if (S.snapshot) { S.snapshot.phase = payload.phase; renderStepper(S.snapshot); }
      break;

    case "step_end":
      closeStreams();
      break;

    case "question": {
      closeStreams();
      stopActivityIndicator();
      const { el, body } = card({ role: "interview", kind: "question", time: event.ts,
        tags: [{ text: payload.subject }, { text: `${payload.purpose || "agenda"}` }] });
      body.innerHTML = `<p><b>${escapeHtml(payload.question)}</b></p>` +
        (payload.rationale ? `<p class="muted small">Why: ${escapeHtml(payload.rationale)}</p>` : "") +
        (payload.progress ? `<div class="progress-bar"><i style="width:${payload.progress.percent}%"></i></div>
            <div class="muted small">${payload.progress.covered}/${payload.progress.subjects} subjects covered</div>` : "");
      pushCard(el);
      if (S.snapshot) {
        S.snapshot.awaiting = "answer";
        S.snapshot.pending_question = { id: payload.id, subject: payload.subject, text: payload.question,
          rationale: payload.rationale, options: payload.options || [], purpose: payload.purpose };
        renderQuestionCard(S.snapshot);
      }
      return;
    }

    case "answer": {
      if (payload.channel === "chat") break;   // chat replies are rendered from the response
      const answerTags = [{ text: payload.subject || "" }];
      if (payload.source === "simulated") answerTags.push({ text: "simulated stakeholder", cls: "warn" });
      const { el, body } = card({ role: "user", kind: "user", time: event.ts, tags: answerTags });
      body.innerHTML = `<p>${escapeHtml(payload.answer || "")}</p>`;
      pushCard(el);
      if (S.snapshot) {
        S.snapshot.awaiting = "none";
        S.snapshot.pending_question = null;
        renderQuestionCard(S.snapshot);
      }
      break;
    }

    case "ledger": {
      const added = payload.added || [];
      if (!added.length) break;
      const { el, body } = card({ role: "triage", time: event.ts, tags: [{ text: `${added.length} new`, cls: "ok" }] });
      body.innerHTML = `<ul>${added.slice(0, 8).map((a) =>
        `<li><b>${escapeHtml(a.id)}</b> <span class="muted">${escapeHtml(a.kind)}</span> — ${escapeHtml(a.title || a.body)}</li>`).join("")}</ul>`;
      pushCard(el);
      break;
    }

    case "spec": {
      if (payload.kind === "section" && payload.markdown) {
        finalizeStream("writer", `write:${payload.section_id}`);
        const { el, body } = card({ role: "writer", time: event.ts, tags: [{ text: payload.title || payload.section_id }] });
        body.innerHTML = safeHTML(payload.markdown.slice(0, 4000));
        pushCard(el);
      } else if (payload.kind === "document" && S.snapshot) {
        S.snapshot.spec_markdown = payload.markdown;
        if (payload.stats) renderMetaFromStats(payload.stats);
      } else if (payload.kind === "summary" && S.snapshot) {
        S.snapshot.summary_markdown = payload.markdown;
      }
      if (S.snapshot) renderSpec(S.snapshot);
      return;
    }

    case "diagram": {
      finalizeStream("diagrammer", payload.step || "");
      if (payload.code && S.snapshot) {
        const list = S.diagrams.slice();
        const index = list.findIndex((d) => d.title === payload.title);
        const item = { title: payload.title || "Diagram", code: payload.code, valid: payload.valid !== false };
        if (index >= 0) list[index] = item; else list.push(item);
        S.diagrams = list.slice(-3);
        renderDiagrams(S.diagrams, payload.valid === false ? "syntax problems detected" : "");
      }
      break;
    }

    case "review": {
      const status = payload.status || "";
      const kind = status === "complete" ? "" : (status === "needs_user" ? "warning" : "");
      const tags = [{ text: `round ${payload.round || 1}` },
        { text: status, cls: status === "complete" ? "ok" : (status === "needs_user" ? "warn" : "") }];
      const { el, body } = card({ role: "critic", kind, time: event.ts, tags });
      const issues = payload.issues || [];
      body.innerHTML = `<p>${escapeHtml(payload.summary || "")}</p>` + (issues.length
        ? `<ul>${issues.slice(0, 10).map((i) => `<li><span class="tag ${i.severity === "blocker" ? "err" : ""}">${escapeHtml(i.kind)}/${escapeHtml(i.severity)}</span> ${escapeHtml(i.description)}${i.question_for_user ? ` <span class="muted">→ ${escapeHtml(i.question_for_user)}</span>` : ""}</li>`).join("")}</ul>`
        : "");
      pushCard(el);
      break;
    }

    case "approval": {
      if (payload.requested) {
        stopActivityIndicator();
        if (S.snapshot) {
          S.snapshot.awaiting = "approval";
          S.snapshot.pending_question = null;
          renderQuestionCard(S.snapshot);
        }
        const { el, body } = card({ role: "summary", kind: "question", time: event.ts, tags: [{ text: "approval needed" }] });
        body.innerHTML = '<p>The requirements summary is ready. Review it in the Spec panel and use the approval controls below.</p>';
        pushCard(el);
      } else {
        if (S.snapshot) {
          S.snapshot.awaiting = "none";
          renderQuestionCard(S.snapshot);
        }
        const { el, body } = card({ role: "user", kind: "user", time: event.ts, tags: [{ text: payload.approved ? "approved" : "changes requested" }] });
        body.innerHTML = `<p>${payload.approved ? "Summary approved." : escapeHtml(payload.notes || "Changes requested.")}</p>`;
        pushCard(el);
      }
      break;
    }

    case "tool_call": {
      const { el, body } = card({ role: payload.role || "chat", time: event.ts, tags: [{ text: `tool · ${payload.tool}` }, { text: payload.ok ? "ok" : "failed", cls: payload.ok ? "" : "err" }] });
      const details = document.createElement("details");
      details.className = "tool";
      details.innerHTML = `<summary>${escapeHtml(payload.tool)}(${Object.keys(payload.args || {}).length} args) · ${payload.latency_ms || 0} ms</summary>
        <pre>${escapeHtml(JSON.stringify(payload.args || {}, null, 2))}\n\n${escapeHtml(payload.preview || "")}</pre>`;
      body.appendChild(details);
      pushCard(el);
      break;
    }

    case "artifact": {
      const { el, body } = card({ role: "engine", time: event.ts, tags: [{ text: payload.name, cls: "ok" }] });
      body.innerHTML = `<p><a href="${payload.url}" download>${escapeHtml(payload.name)}</a> · ${bytes(payload.bytes)}</p>`;
      pushCard(el);
      await refreshArtifacts();
      break;
    }

    case "warning": {
      const { el, body } = card({ role: payload.role || "engine", kind: "warning", time: event.ts, tags: [{ text: "warning", cls: "warn" }] });
      body.innerHTML = `<p>${escapeHtml(payload.message || "")}</p>` +
        (payload.detail ? `<details class="tool"><summary>detail</summary><pre>${escapeHtml(payload.detail)}</pre></details>` : "");
      pushCard(el);
      break;
    }

    case "error": {
      stopActivityIndicator();
      closeStreams();
      const { el, body } = card({ role: payload.role || "engine", kind: "error", time: event.ts, tags: [{ text: "error", cls: "err" }] });
      body.innerHTML = `<p>${escapeHtml(payload.message || "unknown error")}</p>`;
      pushCard(el);
      if (S.snapshot) { S.snapshot.status = "error"; S.snapshot.error = payload.message; renderHeader(S.snapshot); }
      break;
    }

    case "session": {
      if (payload.event === "plan_ready") {
        const { el, body } = card({ role: "plan", time: event.ts, tags: [{ text: "interview plan" }] });
        body.innerHTML = `<p><b>${escapeHtml(payload.project_name || "")}</b></p>` +
          (payload.objective ? `<p class="muted small">${escapeHtml(payload.objective)}</p>` : "") +
          `<ol>${(payload.subjects || []).map((s) => `<li>${escapeHtml(s.name)} <span class="tag">${escapeHtml(s.priority)}</span></li>`).join("")}</ol>`;
        pushCard(el);
      } else if (payload.event === "completed") {
        const stats = payload.stats || {};
        const { el, body } = card({ role: "engine", kind: "question", time: event.ts, tags: [{ text: "delivered", cls: "ok" }] });
        body.innerHTML = `<p>Deliverables written.</p><ul>
            <li>${stats.words || 0} words · ${stats.functional || 0} functional / ${stats.non_functional || 0} non-functional requirements</li>
            <li>${(payload.artifacts || []).map((a) => escapeHtml(a.name)).join(", ")}</li></ul>`;
        pushCard(el);
      } else if (payload.event === "transcript_mode") {
        // Safe: subject names are escapeHtml'd, counts are server-side integers.
        const { el, body } = card({ role: "engine", kind: "question", time: event.ts, tags: [{ text: "transcript", cls: "ok" }] });
        const covered = (payload.covered || []).map((name) => `<li>${escapeHtml(name)} <span class="tag">covered</span></li>`);
        const uncovered = (payload.uncovered || []).map((name) => `<li>${escapeHtml(name)} <span class="tag">needs questions</span></li>`);
        body.innerHTML =
          `<p>${escapeHtml(payload.message || "")}</p>` +
          `<ul>${covered.join("")}${uncovered.join("")}</ul>` +
          `<p class="muted small">${payload.excerpts_triaged || 0} excerpt(s) triaged, ` +
          `${payload.ledger_entries || 0} ledger entries seeded</p>`;
        pushCard(el);
      } else if (payload.event === "chat_reply") {
        const { el, body } = card({ role: "chat", time: event.ts, tags: [{ text: `${payload.iterations || 1} step(s)` }] });
        body.innerHTML = safeHTML(payload.text || "");
        pushCard(el);
      } else if (payload.event === "cancelled") {
        const { el, body } = card({ role: "engine", kind: "warning", time: event.ts, tags: [{ text: "stopped" }] });
        body.innerHTML = "<p>Run stopped. Press Resume to continue.</p>";
        pushCard(el);
      }
      await refreshSnapshot();
      break;
    }

    default:
      break;
  }
}

function renderMetaFromStats(stats) {
  if (!S.snapshot) return;
  S.snapshot.requirements_total = stats.requirements_total ?? S.snapshot.requirements_total;
  $("#session-meta").dataset.stats = JSON.stringify(stats);
}

/* ------------------------------------------------------------------ sessions */

const SESSION_PAGE = 20;
const EVENT_PAGE_SIZE = 500;

async function loadSessions() {
  const response = await fetch("/api/sessions?limit=" + SESSION_PAGE + "&offset=" + S.sessionOffset);
  if (!response.ok) throw new Error("session list failed: " + response.status);
  const rows = await response.json();
  const total = parseInt(response.headers.get("X-Total-Count") || String(rows.length), 10);
  S.sessionTotal = total;

  const list = $("#session-list");
  list.innerHTML = "";
  if (!rows.length) { list.innerHTML = '<li class="empty">none yet</li>'; return; }
  rows.forEach((row) => {
    const li = document.createElement("li");
    li.className = "session-item" + (row.id === S.sessionId ? " active" : "");
    const pillClass = row.status === "error" ? "error" : (row.phase === "done" ? "done" : "");
    li.innerHTML = `<span class="t">${escapeHtml(row.title || row.id)}${row.busy ? ' <span class="busy-dot" title="run in progress">●</span>' : ""}</span>
      <span class="m"><span class="phase-pill ${pillClass}">${escapeHtml(row.phase)}</span>${escapeHtml((row.updated_at || "").slice(0, 16).replace("T", " "))}
        <button class="icon-btn" data-action="rename" title="Rename" aria-label="Rename session">&#9998;</button>
        <button class="icon-btn danger" data-action="delete" title="Delete" aria-label="Delete session">&#10005;</button>
      </span>`;
    li.querySelector(".t").addEventListener("click", () => openSession(row.id));
    li.querySelector('[data-action="rename"]').addEventListener("click", (event) => {
      event.stopPropagation(); renameSession(row.id, row.title);
    });
    li.querySelector('[data-action="delete"]').addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm(`Delete "${row.title || row.id}" and its outputs?`)) return;
      try {
        await api(`/api/sessions/${row.id}`, { method: "DELETE" });
        if (S.sessionId === row.id) S.sessionId = null;
        toast("Session deleted");
        await loadSessions();
      } catch (error) { toast(error.message, true); }
    });
    list.appendChild(li);
  });

  const pager = $("#session-pager");
  if (pager) {
    pager.innerHTML = "";
    if (total > SESSION_PAGE) {
      pager.hidden = false;
      pager.textContent = `${rows.length} of ${total} sessions · `;
      const older = document.createElement("button");
      older.className = "btn btn-ghost btn-sm";
      older.textContent = "Load older";
      older.addEventListener("click", async () => {
        S.sessionOffset += SESSION_PAGE;
        try { await loadSessions(); } catch (error) { toast(error.message, true); }
      });
      pager.appendChild(older);
    } else { pager.hidden = true; }
  }
}

async function renameSession(sessionId, currentTitle) {
  const title = prompt("Session title", currentTitle || "");
  if (!title || title === currentTitle) return;
  try {
    await api(`/api/sessions/${sessionId}`, { method: "PATCH", body: { title } });
    toast("Renamed");
    await loadSessions();
  } catch (error) { toast(error.message, true); }
}

async function openSession(sessionId) {
  disconnectStream();
  S.sessionId = sessionId;
  S.lastEventId = 0;
  S.renderedIds = new Set();
  S.openStreams.clear();
  S.diagrams = [];
  $("#timeline").innerHTML = "";
  $("#tab-trace").innerHTML = "";

  await refreshSnapshot();
  let cursor = 0;
  while (true) {
    const events = await api(`/api/sessions/${sessionId}/events?after=${cursor}&limit=${EVENT_PAGE_SIZE}`);
    for (const event of events) {
      await handleEvent(event);
      cursor = Math.max(cursor, Number(event.id) || 0);
    }
    if (events.length < EVENT_PAGE_SIZE) break;
  }
  if (!$("#timeline").children.length) {
    $("#timeline").innerHTML = '<p class="muted">No activity recorded yet.</p>';
  }
  connectStream(sessionId);
  await loadSessions();
}

async function refreshSnapshot() {
  if (!S.sessionId) return null;
  const snapshot = await api(`/api/sessions/${S.sessionId}`);
  S.snapshot = snapshot;
  renderHeader(snapshot);
  renderStepper(snapshot);
  renderQuestionCard(snapshot);
  renderSpec(snapshot);
  renderLedger(snapshot);
  renderArtifacts(snapshot);
  if (snapshot.diagrams && snapshot.diagrams.length) {
    S.diagrams = snapshot.diagrams;
    renderDiagrams(S.diagrams, snapshot.diagrams.some((d) => !d.valid) ? "some syntax warnings" : "");
  } else {
    renderDiagrams([]);
  }
  return snapshot;
}

async function refreshArtifacts() {
  if (!S.sessionId) return;
  try { await refreshSnapshot(); } catch (_) { /* non fatal */ }
}

/* ------------------------------------------------------------------ commands */

async function submitAnswer({ answer = null, skipped = false, notApplicable = false }) {
  const question = S.snapshot && S.snapshot.pending_question;
  if (!question) { toast("There is no pending question right now.", true); return; }
  try {
    const result = await api(`/api/sessions/${S.sessionId}/answer`, {
      method: "POST", body: { question_id: question.id, answer, skipped, not_applicable: notApplicable },
    });
    if (result && result.ok === false) toast(result.error || "could not submit", true);
    else $("#answer-input").value = "";
  } catch (error) { toast(error.message, true); }
}

async function approve(approved, notes) {
  try {
    const result = await api(`/api/sessions/${S.sessionId}/approve`, { method: "POST", body: { approved, notes } });
    if (result && result.ok === false) toast(result.error || "could not submit approval", true);
  } catch (error) { toast(error.message, true); }
}

async function sendChat(message) {
  // Chat runs in the background; disable the composer so a second message cannot race the first.
  const input = $("#chat-input");
  const button = $("#chat-form button[type=submit]");
  input.disabled = true;
  button.disabled = true;
  button.textContent = "Working…";
  const { el, body } = card({ role: "user", kind: "user", tags: [{ text: "you" }] });
  body.innerHTML = `<p>${escapeHtml(message)}</p>`;
  pushCard(el);
  const pending = card({ role: "chat", tags: [{ text: "thinking…" }] });
  pending.body.innerHTML = '<span class="muted">looking into it…</span>';
  pushCard(pending.el);
  try {
    await api(`/api/sessions/${S.sessionId}/chat`, { method: "POST", body: { message } });
    pending.el.remove();          // the persisted chat_reply event renders the answer
  } catch (error) {
    pending.body.innerHTML = `<p style="color:var(--error)">${escapeHtml(error.message)}</p>`;
  } finally {
    input.disabled = false;
    button.disabled = false;
    button.textContent = "Ask";
  }
}

/* -------------------------------------------------------------------- models */

async function loadModels() {
  let data;
  try {
    data = await api("/api/models");
  } catch (error) {
    $("#health-dot").className = "dot dot-off";
    $("#health-text").textContent = (S.provider || "LM Studio") + ": unreachable";
    return;
  }
  S.models = data.chat_models || [];
  S.defaultModel = data.default_model;
  S.overrides = data.overrides || {};
  S.embeddingModel = data.embedding_model;
  S.provider = data.provider || "LM Studio";

  const models = data.chat_models || [];
  const healthy = models.length > 0;
  $("#health-dot").className = "dot " + (healthy ? "dot-on" : "dot-off");
  // Only providers with an admin API report load state; Strata does not, so "0 loaded" there would
  // be a lie — the server only lists models it can actually serve.
  const reportsState = models.some((m) => m.state === "loaded" || m.state === "not loaded");
  const loaded = models.filter((m) => m.state === "loaded").length;
  const providerName = data.provider || "LM Studio";
  $("#health-text").textContent = !healthy
    ? `${providerName}: no models found`
    : `${providerName} · ${models.length} model(s)${reportsState ? `, ${loaded} loaded` : ""}`;

  renderProviderSelect(data);
  fillModelSelect($("#model-select"), S.defaultModel, true);
  fillModelSelect($("#new-model"), S.defaultModel, true);
  fillEmbeddingSelect(data.embedding_models || []);
  buildRoleGrid();
}

function modelLabel(model) {
  const bits = [];
  if (model.state === "loaded") bits.push("● loaded");
  if (model.capabilities && model.capabilities.includes("tool_use")) bits.push("tools");
  if (model.quantization) bits.push(model.quantization);
  if (model.context_length || model.max_context_length) bits.push(`${Math.round((model.loaded_context_length || model.max_context_length) / 1024)}k ctx`);
  return `${model.id}${bits.length ? " — " + bits.join(", ") : ""}`;
}

function fillModelSelect(select, selected, includeDefault, defaultLabel = "(default / first loaded model)") {
  if (!select) return;
  select.innerHTML = "";
  if (includeDefault) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = defaultLabel;
    select.appendChild(option);
  }
  S.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = modelLabel(model);
    select.appendChild(option);
  });
  if (selected) select.value = selected;
}

function renderProviderSelect(models) {
  const select = $("#provider-select");
  if (!select) return;
  const names = { lmstudio: "LM Studio", strata: "Strata" };
  select.innerHTML = (models.providers || ["lmstudio", "strata"])
    .map((p) => `<option value="${escapeHtml(p)}"${p === models.provider ? " selected" : ""}>${escapeHtml(names[p] || p)}</option>`)
    .join("");
  const base = $("#provider-base");
  if (base) base.textContent = models.base_url ? `${models.provider} → ${models.base_url}` : "";
}

async function saveProvider() {
  const select = $("#provider-select");
  if (!select) return;
  try {
    const saved = await api("/api/models", { method: "POST", body: { provider: select.value } });
    S.provider = saved.provider;
    toast(`Backend switched to ${saved.provider}`);
    await loadModels();
    await loadHealth();
  } catch (error) {
    toast(`Provider switch failed: ${error.message}`, true);
  }
}

function fillEmbeddingSelect(embeddings) {
  const select = $("#embedding-select");
  if (!select) return;
  select.innerHTML = '<option value="">(none — BM25 keyword search only)</option>' +
    embeddings.map((m) => `<option value="${escapeHtml(m.id)}"${m.id === S.embeddingModel ? " selected" : ""}>${escapeHtml(m.id)}</option>`).join("");
}

const SETTING_LABELS = {
  max_questions_per_subject: "Questions per subject",
  min_subjects: "Minimum subjects",
  max_subjects: "Maximum subjects",
  max_gap_rounds: "Gap rounds",
  max_gap_questions_per_round: "Gap questions per round",
  max_summary_rejections: "Summary rejections allowed",
  max_total_questions: "Total questions ceiling",
  max_transcript_chunks: "Transcript chunks",
  transcript_chunk_chars: "Transcript chunk size",
  max_review_rounds: "Review rounds",
  min_functional_requirements: "Minimum requirements",
  require_acceptance_criteria: "Require acceptance criteria",
  max_revision_rounds: "Diagram revision rounds",
  max_diagrams: "Maximum diagrams",
  local_renderer: "Local Mermaid renderer",
  docs_dir: "Knowledge documents directory",
  chunk_size: "Knowledge chunk size",
  chunk_overlap: "Chunk overlap",
  heading_aware: "Heading-aware chunking",
  top_k: "Retrieval top-k",
  min_score: "Minimum score",
  subject_coverage_threshold: "Transcript coverage needed to skip a question",
  render_timeout_seconds: "Mermaid render timeout (s)",
  emit_png: "Also save a PNG of each diagram",
};

async function loadSettings() {
  try {
    const data = await api("/api/settings");
    S.settings = data.settings;
    renderSettingsGrid(data.settings);

    renderProviderSelect(data.models || {});

    // Model lanes live in the same dialog: budgets and model choice are both "how this run behaves".
    const lanes = $("#settings-models");
    if (lanes) {
      const overrides = data.models?.overrides || {};
      lanes.innerHTML = (data.models?.roles || []).map((role) => `
        <label>${escapeHtml(ROLE_LABELS[role] || role)}
          <select id="settings-role-${role}" data-role="${role}">
            <option value="">${escapeHtml(role === "human" ? "Real human" : (data.models?.default_model || "provider default"))}</option>
            ${(S.models || []).map((m) => `<option value="${escapeHtml(m.id)}" ${overrides[role] === m.id ? "selected" : ""}>${escapeHtml(modelLabel(m))}</option>`).join("")}
          </select></label>`).join("");
    }

    // What previous runs taught the engine — the feedback loop should be visible, not hidden in a table.
    // Values come from our own SQLite store and every field is HTML-escaped before insertion.
    const lessonsBox = $("#settings-lessons");
    if (lessonsBox) {
      const lessons = await api("/api/lessons?limit=12");
      lessonsBox.innerHTML = lessons.lessons.length
        ? lessons.lessons
            .map(
              (l) =>
                `<li><span class="lesson-kind">${escapeHtml(l.kind)}</span> ${escapeHtml(l.text)}` +
                (l.seen_count > 1 ? ` <span class="lesson-seen">seen ${l.seen_count}×</span>` : "") +
                `</li>`
            )
            .join("")
        : "<li>No lessons recorded yet — they appear after a run completes.</li>";
    }
  } catch (error) { toast(error.message, true); }
}

// Budget fields only. The store keeps a delta over config.yaml containing just the keys the user changed,
// so this grid always reflects the effective values rather than a frozen snapshot.
function renderSettingsGrid(settings) {
  const grid = $("#settings-grid");
  if (!grid) return;
  grid.innerHTML = Object.entries(settings).map(([section, values]) => `
      <div class="settings-section">
        <h3>${escapeHtml(section)}</h3>
        ${Object.entries(values).map(([key, value]) => `
          <label>${escapeHtml(SETTING_LABELS[key] || key)}
            ${typeof value === "boolean"
              ? `<select data-section="${section}" data-key="${key}">
                   <option value="true" ${value ? "selected" : ""}>yes</option>
                   <option value="false" ${value ? "" : "selected"}>no</option>
                 </select>`
              : typeof value === "number"
                ? `<input type="number" data-section="${section}" data-key="${key}" value="${value}">`
                : `<input type="text" data-section="${section}" data-key="${key}" value="${escapeHtml(String(value))}">`}
          </label>`).join("")}
      </div>`).join("");
}

function buildRoleGrid() {
  const grid = $("#role-grid");
  if (!grid) return;
  const roles = ["plan", "interview", "triage", "summary", "writer", "use_cases", "glossary", "critic", "diagrammer", "mermaid_reviewer", "human"];
  grid.innerHTML = roles.map((role) => `
    <label>${escapeHtml(ROLE_LABELS[role] || role)}
      <select data-role="${role}"></select></label>`).join("");
  grid.querySelectorAll("select").forEach((select) => {
    const role = select.dataset.role;
    fillModelSelect(select, S.overrides[role] || "", true, role === "human" ? "Real human" : undefined);
  });
}

async function saveModels() {
  const defaultModel = $("#model-select").value || null;
  try {
    await api("/api/models", { method: "POST", body: { default_model: defaultModel } });
    S.defaultModel = defaultModel;
    toast("Default model updated");
  } catch (error) { toast(error.message, true); }
}

async function saveRoleOverrides() {
  const overrides = {};
  $$("#role-grid select").forEach((select) => { if (select.value) overrides[select.dataset.role] = select.value; });
  // The select is sent too: choosing "" is a deliberate "no embeddings, BM25 only" — a value the server
  // cannot infer from an absent field.
  const embeddingSelect = $("#embedding-select");
  const embedding = embeddingSelect ? embeddingSelect.value : null;
  try {
    const saved = await api("/api/models", {
      method: "POST",
      body: { overrides, ...(embeddingSelect ? { embedding_model: embedding } : {}) },
    });
    S.overrides = overrides;
    S.embeddingModel = saved.embedding_model || null;
    toast("Per-role models saved");
    loadHealth();
  } catch (error) { toast(error.message, true); }
}

/* -------------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const data = await api("/api/health");
    const kb = data.knowledge || {};
    // Be explicit about which retrieval is actually running: without an embedding model the search is
    // lexical (BM25), not semantic, and the provider determines whether embeddings are even available.
    const retrieval = kb.vector_search
      ? `BM25 + vector (<b>${escapeHtml(kb.embedding_model || "unknown")}</b>)`
      : `<b>BM25 keyword only</b> — ${escapeHtml(kb.vector_search_reason || "no embedding model")}`;
    const pdf = data.pdf || {};
    const engine = pdf.engine
      ? `${escapeHtml(pdf.engine)}${pdf.engine === "typst" ? "" : " (Typst binary not found)"}`
      : "none";
    $("#kb-status").innerHTML =
      `${kb.chunks || 0} chunks from ${kb.document_count || 0} document(s)<br>retrieval: ${retrieval}<br>PDF: ${engine}`;
  } catch (error) {
    $("#kb-status").textContent = "knowledge base unavailable";
  }
}

/* ------------------------------------------------------------------ wiring */

function wire() {
  // tabs
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $$(".tab-pane").forEach((pane) => pane.classList.toggle("active", pane.id === `tab-${tab.dataset.tab}`));
  }));

  // new session
  $("#btn-new").addEventListener("click", () => { fillModelSelect($("#new-model"), S.defaultModel, true); $("#dlg-new").showModal(); });
  $("#new-form").addEventListener("submit", async (event) => {
    if (event.submitter && event.submitter.value !== "create") return;
    event.preventDefault();
    const brief = $("#new-brief").value.trim();
    const transcript = $("#new-transcript").value.trim();
    if (!brief && !transcript) { toast("Give a project brief, or paste a transcript to start from.", true); return; }
    const urls = $("#new-urls").value.split("\n").map((u) => u.trim()).filter(Boolean);
    const button = $("#new-create");
    button.disabled = true;
    try {
      const overrides = {};
      if ($("#new-model").value) overrides.default = $("#new-model").value;
      const data = await api("/api/sessions", {
        method: "POST",
        body: {
          brief,
          title: $("#new-title").value.trim(),
          urls,
          model_overrides: overrides,
          transcript,
          transcript_source: transcript ? (S.transcriptSource || "pasted transcript") : "",
        },
      });
      $("#dlg-new").close();
      $("#new-brief").value = ""; $("#new-urls").value = ""; $("#new-title").value = "";
      $("#new-transcript").value = "";
      S.transcriptSource = "";
      updateTranscriptCount();
      await openSession(data.session.id);
    } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
  });

  const updateTranscriptCount = () => {
    const count = $("#transcript-count");
    if (!count) return;
    const length = $("#new-transcript").value.length;
    count.textContent = length ? `${length.toLocaleString()} characters loaded` : "";
  };
  $("#new-transcript").addEventListener("input", updateTranscriptCount);
  $("#new-transcript-file-btn").addEventListener("click", () => $("#new-transcript-file").click());
  $("#new-transcript-file").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    if (file.size > 200000) {
      toast("That transcript is larger than 200 KB — paste the relevant part instead.", true);
      return;
    }
    file.text().then((text) => {
      $("#new-transcript").value = text;
      S.transcriptSource = file.name;
      updateTranscriptCount();
      toast(`Loaded ${file.name} (${text.length.toLocaleString()} characters).`);
    });
  });

  // answers
  $("#answer-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const value = $("#answer-input").value.trim();
    if (!value) { toast("Type an answer, or use Skip / Not applicable.", true); return; }
    submitAnswer({ answer: value });
  });
  $$("[data-skip]").forEach((button) => button.addEventListener("click", () => {
    submitAnswer({ skipped: button.dataset.skip === "skip", notApplicable: button.dataset.skip === "na" });
  }));
  $("#answer-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      const value = event.target.value.trim();
      if (value) submitAnswer({ answer: value });
    }
  });

  // chat
  $("#chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#chat-input");
    const message = input.value.trim();
    if (!message || !S.sessionId) return;
    input.value = "";
    await sendChat(message);
  });

  // reject dialog
  $("#reject-form").addEventListener("submit", (event) => {
    if (event.submitter && event.submitter.value !== "send") return;
    event.preventDefault();
    approve(false, $("#reject-notes").value.trim());
    $("#dlg-reject").close();
  });

  // models
  $("#model-select").addEventListener("change", saveModels);
  $("#provider-select")?.addEventListener("change", saveProvider);
  $("#btn-advanced").addEventListener("click", () => { buildRoleGrid(); $("#dlg-advanced").showModal(); });
  $("#btn-settings").addEventListener("click", () => { loadSettings(); $("#dlg-settings").showModal(); });
  $("#settings-reset")?.addEventListener("click", async () => {
    // The store keeps only the keys the user changed; resetting clears that delta so config.yaml wins again.
    try {
      const data = await api("/api/settings/reset", { method: "POST" });
      S.settings = data.settings;
      renderSettingsGrid(data.settings);
      toast("Settings reset to the values this server started with");
    } catch (error) {
      toast(`Reset failed: ${error.message}`);
    }
  });

  $("#settings-form").addEventListener("submit", async (event) => {
    if (event.submitter && event.submitter.value !== "save") return;
    event.preventDefault();
    const patch = {};
    $$("#settings-grid [data-section][data-key]").forEach((field) => {
      const section = field.dataset.section;
      const key = field.dataset.key;
      const raw = field.value.trim();
      if (raw === "") return;
      const current = S.settings[section][key];
      patch[section] = Object.assign(patch[section] || {}, {
        [key]: typeof current === "number" ? Number(raw) : typeof current === "boolean" ? raw === "true" : raw,
      });
    });
    const laneSelects = $$("#settings-models select");
    const lanes = Object.fromEntries(laneSelects.map((select) => [select.dataset.role, select.value]));
    if (!Object.keys(patch).length && !laneSelects.length) { toast("Nothing to apply."); return; }
    try {
      if (Object.keys(patch).length) await api("/api/settings", { method: "PATCH", body: patch });
      if (laneSelects.length) {
        await api("/api/models", { method: "POST", body: { overrides: lanes } });
        S.overrides = Object.fromEntries(Object.entries(lanes).filter(([, model]) => model));
      }
      toast("Settings updated.");
      $("#dlg-settings").close();
    } catch (error) { toast(error.message, true); }
  });
  $("#adv-form").addEventListener("submit", async (event) => {
    if (event.submitter && event.submitter.value !== "save") return;
    event.preventDefault();
    await saveRoleOverrides();
    $("#dlg-advanced").close();
  });

  // run controls
  $("#btn-stop").addEventListener("click", async () => {
    try { await api(`/api/sessions/${S.sessionId}/cancel`, { method: "POST" }); toast("Stop requested"); }
    catch (error) { toast(error.message, true); }
  });
  $("#btn-resume").addEventListener("click", async () => {
    try { await api(`/api/sessions/${S.sessionId}/resume`, { method: "POST" }); toast("Resumed"); }
    catch (error) { toast(error.message, true); }
  });

  // knowledge base
  $("#kb-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const files = $("#kb-files").files;
    if (!files.length) { toast("Choose files first.", true); return; }
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    try {
      const data = await api("/api/knowledge/upload", { method: "POST", body: form });
      const failed = (data.results || []).filter((r) => !r.ok);
      toast(failed.length ? `Indexed with ${failed.length} failure(s)` : "Knowledge base updated");
      await loadHealth();
    } catch (error) { toast(error.message, true); }
  });
  $("#kb-index-dir").addEventListener("click", async () => {
    try {
      const data = await api("/api/knowledge/index", { method: "POST", body: {} });
      const dir = data.directory || {};
      toast(`Indexed ${dir.indexed || 0}, skipped ${dir.skipped || 0}${(dir.failed || []).length ? `, failed ${dir.failed.length}` : ""}`);
      await loadHealth();
    } catch (error) { toast(error.message, true); }
  });

  $("#btn-diagram-refresh").addEventListener("click", () => renderDiagrams(S.diagrams, "re-rendered"));

  const ledgerDialog = $("#dlg-ledger");
  if (ledgerDialog) {
    ledgerDialog.addEventListener("submit", async (event) => {
      if (event.submitter && event.submitter.value !== "save") return;
      event.preventDefault();
      const itemId = ledgerDialog.dataset.itemId;
      if (!itemId) return;
      try {
        await api(`/api/sessions/${S.sessionId}/ledger/${encodeURIComponent(itemId)}`, {
          method: "POST",
          body: {
            title: $("#ledger-title").value,
            body: $("#ledger-body").value,
            priority: $("#ledger-priority").value,
            acceptance: $("#ledger-acceptance").value.split("\n").map((s) => s.trim()).filter(Boolean),
          },
        });
        ledgerDialog.close();
        toast("Ledger updated — the spec will be rebuilt from it");
        await refreshSnapshot();
      } catch (error) { toast(error.message, true); }
    });
  }

  window.addEventListener("beforeunload", disconnectStream);
}

async function init() {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "strict", flowchart: { htmlLabels: true } });
  }
  wire();
  await Promise.all([loadModels(), loadHealth(), loadSessions()]);
}

document.addEventListener("DOMContentLoaded", init);
