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
  mermaid_reviewer: "Mermaid reviewer", chat: "BA assistant", engine: "Workflow",
};

const S = {
  sessionId: null,
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
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ utilities */

async function api(path, options = {}) {
  const init = { headers: {}, ...options };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
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

function pushCard(el) {
  const timeline = $("#timeline");
  const welcome = timeline.querySelector(".welcome");
  if (welcome) welcome.remove();
  timeline.appendChild(el);
  timeline.scrollTop = timeline.scrollHeight;
}

function closeStreams() { S.openStreams.clear(); }

function appendToken(role, step, text) {
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
  $("#timeline").scrollTop = $("#timeline").scrollHeight;
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

function traceLine(event) {
  const pane = $("#tab-trace");
  const line = document.createElement("div");
  line.className = "trace-line";
  const payload = event.payload || {};
  const extra = payload.phase || payload.section_id || payload.name || payload.tool || payload.model || "";
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

function renderHeader(snapshot) {
  if (!snapshot) return;
  $("#session-title").textContent = snapshot.project_name || snapshot.title || "Untitled";
  const bits = [];
  if (snapshot.objective) bits.push(snapshot.objective);
  bits.push(`${snapshot.questions_total} questions · ${snapshot.requirements_total} requirements`);
  if (snapshot.metrics && snapshot.metrics.model_seconds) {
    bits.push(`model time ${Math.round(snapshot.metrics.model_seconds)}s / ${Math.round(snapshot.metrics.calls || 0)} calls`);
  }
  $("#session-meta").textContent = bits.join(" · ");
  const waiting = snapshot.awaiting !== "none" && snapshot.phase !== "done";
  $("#composer").hidden = !snapshot;
  $("#btn-stop").hidden = !snapshot.busy;
  // An errored run is resumable by design (e.g. LM Studio was off, or a model finished loading since).
  const resumable = ["idle", "cancelled", "error"].includes(snapshot.status) && snapshot.phase !== "done";
  $("#btn-resume").hidden = !resumable;
  return waiting;
}

function renderQuestionCard(snapshot) {
  const box = $("#question-card");
  const question = snapshot.pending_question;
  if (!question) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
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
  $("#answer-input").focus();
}

function renderSpec(snapshot) {
  const pane = $("#tab-spec");
  if (!snapshot) return;
  const markdown = snapshot.spec_markdown || "";
  if (markdown.trim()) {
    pane.innerHTML = safeHTML(markdown);
  } else if (snapshot.summary_markdown) {
    pane.innerHTML = `<p class="muted">Draft not started yet — approved summary below.</p>` + safeHTML(snapshot.summary_markdown);
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
    </tr>`).join("");

  const otherRows = others.map((i) => `<tr>
      <td><span class="tag">${escapeHtml(i.kind)}</span></td>
      <td>${escapeHtml(i.body || i.title || "")}</td>
      <td class="muted">${escapeHtml(i.subject || "")}</td>
    </tr>`).join("");

  pane.innerHTML = `
    <p class="small muted">${requirements.length} requirement(s) · ${others.length} other ledger entries</p>
    ${requirements.length ? `<table class="data"><thead><tr><th>ID</th><th>Title</th><th>Category</th><th>Priority</th><th>Acceptance</th><th>Source</th></tr></thead><tbody>${reqRows}</tbody></table>` : ""}
    ${otherRows ? `<h3 style="font-size:13px;margin-top:18px">Facts, decisions, risks</h3><table class="data"><tbody>${otherRows}</tbody></table>` : ""}`;
}

function renderArtifacts(snapshot) {
  const pane = $("#tab-artifacts");
  const items = (snapshot && snapshot.artifacts) || [];
  if (!items.length) { pane.innerHTML = '<p class="muted">No artefacts yet.</p>'; return; }
  pane.innerHTML = `<ul class="artifact-list">${items.map((a) => `
    <li><a href="/api/artifacts/${encodeURIComponent(S.sessionId)}/${encodeURIComponent(a.name)}" download>
      <span>${escapeHtml(a.name)}</span><span class="muted small">${bytes(a.bytes)}</span></a></li>`).join("")}</ul>
    <p class="muted small">Files are also on disk under <code>outputs/${escapeHtml(S.sessionId)}/</code>.</p>`;
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

/* --------------------------------------------------------------- event stream */

function connectStream(sessionId) {
  disconnectStream();
  const url = `/api/sessions/${sessionId}/stream?after=${S.lastEventId}`;
  const es = new EventSource(url);
  S.es = es;
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

  switch (type) {
    case "token":
      appendToken(payload.role || "writer", payload.step || "", payload.text || "");
      return;

    case "reasoning":
      appendToken(payload.role || "writer", (payload.step || "") + ":thinking", payload.text || "");
      return;

    case "step_start":
      closeStreams();
      if (S.snapshot) { S.snapshot.phase = payload.phase; renderStepper(S.snapshot); }
      break;

    case "step_end":
      closeStreams();
      break;

    case "question": {
      closeStreams();
      const { el, body } = card({ role: "interview", kind: "question", time: event.ts,
        tags: [{ text: payload.subject }, { text: `${payload.purpose || "agenda"}` }] });
      body.innerHTML = `<p><b>${escapeHtml(payload.question)}</b></p>` +
        (payload.rationale ? `<p class="muted small">Why: ${escapeHtml(payload.rationale)}</p>` : "") +
        (payload.progress ? `<div class="progress-bar"><i style="width:${payload.progress.percent}%"></i></div>
            <div class="muted small">${payload.progress.covered}/${payload.progress.subjects} subjects covered</div>` : "");
      pushCard(el);
      if (S.snapshot) {
        S.snapshot.pending_question = { id: payload.id, subject: payload.subject, text: payload.question,
          rationale: payload.rationale, options: payload.options || [], purpose: payload.purpose };
        renderQuestionCard(S.snapshot);
      }
      return;
    }

    case "answer": {
      if (payload.channel === "chat") break;   // chat replies are rendered from the response
      const { el, body } = card({ role: "user", kind: "user", time: event.ts, tags: [{ text: payload.subject || "" }] });
      body.innerHTML = `<p>${escapeHtml(payload.answer || "")}</p>`;
      pushCard(el);
      if (S.snapshot) { S.snapshot.pending_question = null; renderQuestionCard(S.snapshot); }
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
        const { el, body } = card({ role: "summary", kind: "question", time: event.ts, tags: [{ text: "approval needed" }] });
        body.innerHTML = `<p>The requirements summary is ready for your review — see the <b>Spec</b> panel.</p>
          <div style="display:flex;gap:8px;margin-top:8px">
            <button class="btn btn-primary btn-sm" data-approve="yes">Approve &amp; draft the FSD</button>
            <button class="btn btn-ghost btn-sm" data-approve="no">Request changes…</button></div>`;
        pushCard(el);
        el.querySelector('[data-approve="yes"]').addEventListener("click", () => approve(true, ""));
        el.querySelector('[data-approve="no"]').addEventListener("click", () => $("#dlg-reject").showModal());
      } else {
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

async function loadSessions() {
  const rows = await api("/api/sessions");
  const list = $("#session-list");
  list.innerHTML = "";
  if (!rows.length) { list.innerHTML = '<li class="empty">none yet</li>'; return; }
  rows.forEach((row) => {
    const li = document.createElement("li");
    li.className = "session-item" + (row.id === S.sessionId ? " active" : "");
    const pillClass = row.status === "error" ? "error" : (row.phase === "done" ? "done" : "");
    li.innerHTML = `<span class="t">${escapeHtml(row.title || row.id)}</span>
      <span class="m"><span class="phase-pill ${pillClass}">${escapeHtml(row.phase)}</span>${escapeHtml((row.updated_at || "").slice(0, 16).replace("T", " "))}</span>`;
    li.addEventListener("click", () => openSession(row.id));
    list.appendChild(li);
  });
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
  const events = await api(`/api/sessions/${sessionId}/events?after=0&limit=1500`);
  for (const event of events) await handleEvent(event);
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

  const healthy = (data.chat_models || []).length > 0;
  $("#health-dot").className = "dot " + (healthy ? "dot-on" : "dot-off");
  const loaded = (data.chat_models || []).filter((m) => m.state === "loaded").map((m) => m.id);
  $("#health-text").textContent = healthy
    ? `${data.provider || "LM Studio"} · ${data.chat_models.length} model(s), ${loaded.length} loaded`
    : (data.provider || "LM Studio") + ": no models found";

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

function fillModelSelect(select, selected, includeDefault) {
  if (!select) return;
  select.innerHTML = "";
  if (includeDefault) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "(default / first loaded model)";
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

function fillEmbeddingSelect(embeddings) {
  const select = $("#embedding-select");
  if (!select) return;
  select.innerHTML = '<option value="">(none — BM25 keyword search only)</option>' +
    embeddings.map((m) => `<option value="${escapeHtml(m.id)}"${m.id === S.embeddingModel ? " selected" : ""}>${escapeHtml(m.id)}</option>`).join("");
}

function buildRoleGrid() {
  const grid = $("#role-grid");
  if (!grid) return;
  const roles = ["plan", "interview", "triage", "summary", "writer", "use_cases", "glossary", "critic", "diagrammer", "mermaid_reviewer"];
  grid.innerHTML = roles.map((role) => `
    <label>${escapeHtml(ROLE_LABELS[role] || role)}
      <select data-role="${role}"></select></label>`).join("");
  grid.querySelectorAll("select").forEach((select) => {
    const role = select.dataset.role;
    fillModelSelect(select, S.overrides[role] || "", true);
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
  const embedding = $("#embedding-select").value || null;
  try {
    await api("/api/models", { method: "POST", body: { overrides } });
    S.overrides = overrides;
    toast("Per-role models saved");
  } catch (error) { toast(error.message, true); }
}

/* -------------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const data = await api("/api/health");
    const kb = data.knowledge || {};
    $("#kb-status").innerHTML = `${kb.chunks || 0} chunks from ${kb.documents || 0} document(s)<br>` +
      `vector search: <b>${kb.vector_search ? "on" : "off (BM25 only)"}</b>`;
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
    if (!brief) { toast("A project brief is required.", true); return; }
    const urls = $("#new-urls").value.split("\n").map((u) => u.trim()).filter(Boolean);
    const button = $("#new-create");
    button.disabled = true;
    try {
      const overrides = {};
      if ($("#new-model").value) overrides.default = $("#new-model").value;
      const data = await api("/api/sessions", {
        method: "POST",
        body: { brief, title: $("#new-title").value.trim(), urls, model_overrides: overrides },
      });
      $("#dlg-new").close();
      $("#new-brief").value = ""; $("#new-urls").value = ""; $("#new-title").value = "";
      await openSession(data.session.id);
    } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
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
  $("#btn-advanced").addEventListener("click", () => { buildRoleGrid(); $("#dlg-advanced").showModal(); });
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
