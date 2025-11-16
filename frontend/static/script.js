// CrownTALK EXTREME v3 – merged frontend for current backend
// Backend base (no /api/generate anymore)
const backendBase = "https://crowntalk-v2-0.onrender.com";
const COMMENT_URL = `${backendBase}/comment`;
const REROLL_URL = `${backendBase}/reroll`;

/** State */
let currentController = null;
let cancelled = false;
let clipboardHistory = [];

/** DOM refs */
const els = {
  backendUrl: document.getElementById("ct-backend-url"),
  urlsInput: document.getElementById("ct-urls-input"),
  themeGrid: document.getElementById("ct-theme-grid"),
  btnGenerate: document.getElementById("ct-generate-btn"),
  btnCancel: document.getElementById("ct-cancel-btn"),
  btnClear: document.getElementById("ct-clear-btn"),
  progressFill: document.getElementById("ct-progress-fill"),
  progressLabel: document.getElementById("ct-progress-label"),
  statusPill: document.getElementById("ct-status-pill"),
  skeletonContainer: document.getElementById("ct-skeleton-container"),
  resultsList: document.getElementById("ct-results-list"),
  historyDrawer: document.getElementById("ct-history-drawer"),
  historyBody: document.getElementById("ct-history-body"),
  historyCloseBtn: document.getElementById("ct-history-close-btn"),
  historyClearBtn: document.getElementById("ct-history-clear-btn"),
  historyToggleBtn: document.getElementById("ct-collapse-history-btn"),
};

/* ------------------ Utils ------------------ */

function setStatus(text, type) {
  if (!els.statusPill) return;
  els.statusPill.textContent = text;
  els.statusPill.classList.remove("ct-status-pill--ok", "ct-status-pill--error");
  if (type === "ok") els.statusPill.classList.add("ct-status-pill--ok");
  if (type === "error") els.statusPill.classList.add("ct-status-pill--error");
}

function setProgress(percent, label) {
  if (els.progressFill) {
    const clamped = Math.max(0, Math.min(100, percent));
    els.progressFill.style.width = `${clamped}%`;
  }
  if (label && els.progressLabel) {
    els.progressLabel.textContent = label;
  }
}

function setSkeletonVisible(visible) {
  if (!els.skeletonContainer) return;
  els.skeletonContainer.classList.toggle("is-hidden", !visible);
}

function clearResults() {
  if (els.resultsList) {
    els.resultsList.innerHTML = "";
  }
}

/** Parse URLs: trim, dedupe, ignore empty lines */
function parseUrls(raw) {
  const lines = raw.split(/\r?\n/).map((l) => l.trim());
  const cleaned = [];
  const seen = new Set();
  for (const line of lines) {
    if (!line) continue;
    if (seen.has(line)) continue;
    seen.add(line);
    cleaned.push(line);
  }
  return cleaned;
}

/** Active theme ID from UI (for labels only, backend ignores it) */
function getActiveThemeId() {
  if (!els.themeGrid) return "auto";
  const active = els.themeGrid.querySelector(".ct-theme-btn.is-active");
  return active ? active.dataset.themeId || "auto" : "auto";
}

/** Split "native (english)" into { native, english } */
function splitComment(comment) {
  if (!comment) return { native: "", english: "" };
  const match = comment.match(/^(.*)\(([^)]*)\)\s*$/);
  if (match) {
    const native = match[1].trim();
    const english = match[2].trim();
    if (native && english) {
      return { native, english };
    }
  }
  // fallback: treat whole string as native only
  return { native: comment.trim(), english: "" };
}

/** Enable/disable buttons depending on running state */
function setRunning(isRunning) {
  if (els.btnGenerate) els.btnGenerate.disabled = isRunning;
  if (els.btnCancel) els.btnCancel.disabled = !isRunning;
  if (els.btnClear) els.btnClear.disabled = isRunning;
}

/* ------------------ Clipboard History ------------------ */

function pushClipboardHistory(entry) {
  const ts = new Date();
  clipboardHistory.unshift({
    ...entry,
    time: ts,
  });
  clipboardHistory = clipboardHistory.slice(0, 20);
  renderClipboardHistory();
}

function renderClipboardHistory() {
  if (!els.historyBody) return;
  const container = els.historyBody;
  container.innerHTML = "";

  if (!clipboardHistory.length) {
    const p = document.createElement("p");
    p.className = "ct-history-empty";
    p.textContent = "Nothing copied yet. Copy a comment to see it here.";
    container.appendChild(p);
    return;
  }

  clipboardHistory.forEach((item) => {
    const wrap = document.createElement("div");
    wrap.className = "ct-history-item";

    const header = document.createElement("div");
    header.className = "ct-history-item-header";

    const themeSpan = document.createElement("span");
    themeSpan.className = "ct-history-item-theme";
    themeSpan.textContent = item.label || "Comment";

    const timeSpan = document.createElement("span");
    timeSpan.className = "ct-history-item-time";
    timeSpan.textContent = item.time.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    const body = document.createElement("div");
    body.className = "ct-history-item-body";
    body.textContent = item.text || "";

    header.appendChild(themeSpan);
    header.appendChild(timeSpan);
    wrap.appendChild(header);
    wrap.appendChild(body);
    container.appendChild(wrap);
  });
}

/** Copy helper with status + history */
function copyText(text, label) {
  if (!text) return;
  navigator.clipboard
    .writeText(text)
    .then(() => {
      setStatus("Copied to clipboard", "ok");
      pushClipboardHistory({ text, label });
    })
    .catch(() => {
      setStatus("Copy failed (clipboard blocked)", "error");
    });
}

/* ------------------ History Drawer ------------------ */

function openHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.add("is-open");
}

function closeHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.remove("is-open");
}

/* ------------------ Result Card Rendering ------------------ */

function renderResultCard(item, index, totalCount) {
  if (!els.resultsList) return;

  const card = document.createElement("article");
  card.className = "ct-card";
  fillCardContent(card, item, index, totalCount);
  els.resultsList.appendChild(card);

  // flash highlight on new card
  card.classList.remove("flash-highlight");
  void card.offsetWidth;
  card.classList.add("flash-highlight");
}

function fillCardContent(card, item, index, totalCount) {
  card.innerHTML = "";

  const url = item.url || "";
  const themeId = getActiveThemeId();
  const comments = Array.isArray(item.comments) ? item.comments : [];

  const hasComments = comments.length > 0;

  // Header
  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("a");
  urlEl.className = "ct-card-url";
  urlEl.href = url || "#";
  urlEl.target = "_blank";
  urlEl.rel = "noopener noreferrer";
  urlEl.textContent = url || "(no URL)";
  header.appendChild(urlEl);

  const chipRow = document.createElement("div");
  chipRow.className = "ct-card-chip-row";

  const chipTheme = document.createElement("span");
  chipTheme.className = "ct-chip ct-chip--theme";
  chipTheme.textContent = `Theme: ${themeId}`;
  chipRow.appendChild(chipTheme);

  const chipMood = document.createElement("span");
  chipMood.className = "ct-chip ct-chip--mood";
  chipMood.textContent = "Mode: contextual";
  chipRow.appendChild(chipMood);

  header.appendChild(chipRow);
  card.appendChild(header);

  // Body
  const body = document.createElement("div");
  body.className = "ct-card-body";

  if (!hasComments) {
    const p = document.createElement("p");
    p.className = "ct-comment-text ct-comment-text--faded";
    p.textContent = "No comments generated for this URL.";
    body.appendChild(p);
  } else {
    comments.forEach((rawComment, idx) => {
      const { native, english } = splitComment(rawComment || "");
      const block = document.createElement("div");
      block.className = "ct-comment-block";

      const label = document.createElement("div");
      label.className = "ct-comment-label";
      label.textContent = `Comment ${idx + 1}`;
      block.appendChild(label);

      const nativeP = document.createElement("p");
      nativeP.className = "ct-comment-text";
      nativeP.textContent = native || "(empty)";
      block.appendChild(nativeP);

      if (english) {
        const enP = document.createElement("p");
        enP.className = "ct-comment-text ct-comment-text--faded";
        enP.textContent = english;
        block.appendChild(enP);
      }

      // per-comment copy controls
      const actions = document.createElement("div");
      actions.className = "ct-card-actions";

      const btnNative = document.createElement("button");
      btnNative.className = "ct-card-btn";
      btnNative.textContent = "⧉ Native";
      btnNative.addEventListener("click", () => {
        copyText(native, `Native · C${idx + 1}`);
      });

      const btnEn = document.createElement("button");
      btnEn.className = "ct-card-btn";
      btnEn.textContent = "⧉ EN";
      btnEn.addEventListener("click", () => {
        const text = english || native;
        copyText(text, `English · C${idx + 1}`);
      });

      const btnBoth = document.createElement("button");
      btnBoth.className = "ct-card-btn";
      btnBoth.textContent = "⧉ Both";
      btnBoth.addEventListener("click", () => {
        const parts = [];
        if (native) parts.push(native);
        if (english) parts.push(english);
        const finalText = parts.join("\n\n");
        copyText(finalText, `Both · C${idx + 1}`);
      });

      actions.appendChild(btnNative);
      actions.appendChild(btnEn);
      actions.appendChild(btnBoth);
      block.appendChild(actions);

      body.appendChild(block);
    });
  }

  card.appendChild(body);

  // Footer meta (simple)
  const metaFooter = document.createElement("div");
  metaFooter.className = "ct-meta-footer";

  const metaLeft = document.createElement("span");
  metaLeft.className = "ct-meta-keywords";
  metaLeft.textContent = `Index: ${index + 1}/${totalCount}`;
  const metaRight = document.createElement("span");
  metaRight.textContent = "CrownTALK offline engine";

  metaFooter.appendChild(metaLeft);
  metaFooter.appendChild(metaRight);
  card.appendChild(metaFooter);

  // Card-level actions (reroll)
  const footerActions = document.createElement("div");
  footerActions.className = "ct-card-actions";

  const btnReroll = document.createElement("button");
  btnReroll.className = "ct-card-btn";
  btnReroll.textContent = "🔁 Reroll";
  btnReroll.addEventListener("click", () => {
    if (!url) return;
    rerollSingle(url, card);
  });

  footerActions.appendChild(btnReroll);
  card.appendChild(footerActions);
}

/* ------------------ Reroll ------------------ */

async function rerollSingle(url, cardNode) {
  if (!url) return;

  setStatus("Rerolling…", "ok");
  cardNode.style.opacity = "0.55";

  try {
    const res = await fetch(REROLL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    const data = await res.json();
    if (data && !data.error && Array.isArray(data.comments)) {
      fillCardContent(cardNode, { url: data.url || url, comments: data.comments }, 0, 1);
      setStatus("Reroll done", "ok");
    } else {
      console.error("Reroll failed", data && data.error);
      setStatus("Reroll failed", "error");
    }
  } catch (err) {
    console.error("Reroll network error", err);
    setStatus("Reroll failed (network)", "error");
  } finally {
    cardNode.style.opacity = "1";
  }
}

/* ------------------ Main Generate Flow ------------------ */

async function handleGenerate() {
  const raw = (els.urlsInput && els.urlsInput.value) || "";
  const urls = parseUrls(raw);

  if (!urls.length) {
    setStatus("Paste at least one URL", "error");
    return;
  }

  if (currentController) {
    setStatus("Already running, cancel first", "error");
    return;
  }

  cancelled = false;
  currentController = new AbortController();

  clearResults();
  setSkeletonVisible(true);
  setProgress(5, `Queued ${urls.length} tweets…`);
  setStatus("Contacting backend…", "ok");
  setRunning(true);

  try {
    const res = await fetch(COMMENT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
      signal: currentController.signal,
    });

    if (!res.ok) {
      setSkeletonVisible(false);
      setStatus(`Backend error (HTTP ${res.status})`, "error");
      setProgress(100, "Error");
      return;
    }

    const data = await res.json();
    if (cancelled) {
      setSkeletonVisible(false);
      setStatus("Run cancelled", "error");
      setProgress(0, "Cancelled");
      return;
    }

    const batches = (data && data.batches) || [];
    let totalResults = 0;
    let totalFailed = 0;
    const totalExpected = urls.length || 1;

    setSkeletonVisible(false);
    clearResults();

    batches.forEach((batch, batchIdx) => {
      const batchResults = (batch && batch.results) || [];
      const batchFailed = (batch && batch.failed) || [];

      batchResults.forEach((item, idx) => {
        if (cancelled) return;
        totalResults += 1;
        const percent = 10 + Math.round((totalResults / totalExpected) * 80);
        setProgress(percent, `Rendering ${totalResults}/${totalExpected}…`);
        renderResultCard(item, totalResults - 1, totalExpected);
      });

      batchFailed.forEach((f) => {
        totalFailed += 1;
        renderFailedCard(f);
      });

      if (!cancelled) {
        setStatus(`Batch ${batchIdx + 1} done`, "ok");
      }
    });

    if (!batches.length) {
      setStatus("No valid URLs found", "error");
      setProgress(100, "Done");
    } else {
      setProgress(100, "Completed");
      if (!cancelled) {
        setStatus("Done", "ok");
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Run cancelled", "error");
      setSkeletonVisible(false);
      setProgress(0, "Cancelled");
    } else {
      console.error("Generate failed", err);
      setStatus("Network / server error", "error");
      setSkeletonVisible(false);
      setProgress(100, "Error");
    }
  } finally {
    setRunning(false);
    currentController = null;
  }
}

function renderFailedCard(item) {
  if (!els.resultsList) return;
  const card = document.createElement("article");
  card.className = "ct-card";

  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("a");
  urlEl.className = "ct-card-url";
  urlEl.href = item.url || "#";
  urlEl.target = "_blank";
  urlEl.rel = "noopener noreferrer";
  urlEl.textContent = item.url || "(invalid URL)";
  header.appendChild(urlEl);

  const chipRow = document.createElement("div");
  chipRow.className = "ct-card-chip-row";

  const chipErr = document.createElement("span");
  chipErr.className = "ct-chip";
  chipErr.textContent = "Failed";
  chipRow.appendChild(chipErr);

  header.appendChild(chipRow);
  card.appendChild(header);

  const body = document.createElement("div");
  body.className = "ct-card-body";

  const p = document.createElement("p");
  p.className = "ct-comment-text ct-comment-text--faded";
  p.textContent = item.reason || "Unknown error";
  body.appendChild(p);

  card.appendChild(body);
  els.resultsList.appendChild(card);
}

/* ------------------ Handlers ------------------ */

function handleCancel() {
  if (currentController) {
    cancelled = true;
    currentController.abort();
  }
}

function handleClear() {
  if (els.urlsInput) els.urlsInput.value = "";
  setStatus("Input cleared", "ok");
}

/** Theme click (visual only) */
function handleThemeClick(event) {
  const btn = event.target.closest(".ct-theme-btn");
  if (!btn || !els.themeGrid) return;
  els.themeGrid
    .querySelectorAll(".ct-theme-btn")
    .forEach((b) => b.classList.remove("is-active"));
  btn.classList.add("is-active");
}

/* ------------------ Init ------------------ */

function init() {
  // backend label
  if (els.backendUrl) {
    els.backendUrl.textContent = backendBase;
  }

  // hide language-mode block (you said you don't want it)
  const langOptions = document.getElementById("ct-lang-options");
  if (langOptions) {
    const block = langOptions.closest(".ct-section-block");
    if (block) block.style.display = "none";
  }

  if (els.btnGenerate) {
    els.btnGenerate.addEventListener("click", handleGenerate);
  }
  if (els.btnCancel) {
    els.btnCancel.addEventListener("click", handleCancel);
  }
  if (els.btnClear) {
    els.btnClear.addEventListener("click", handleClear);
  }
  if (els.themeGrid) {
    els.themeGrid.addEventListener("click", handleThemeClick);
  }

  if (els.historyToggleBtn) {
    els.historyToggleBtn.addEventListener("click", () => {
      if (els.historyDrawer && els.historyDrawer.classList.contains("is-open")) {
        closeHistoryDrawer();
      } else {
        openHistoryDrawer();
      }
    });
  }

  if (els.historyCloseBtn) {
    els.historyCloseBtn.addEventListener("click", closeHistoryDrawer);
  }

  if (els.historyClearBtn) {
    els.historyClearBtn.addEventListener("click", () => {
      clipboardHistory = [];
      renderClipboardHistory();
    });
  }

  // ESC closes history
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeHistoryDrawer();
  });

  renderClipboardHistory();
  setStatus("Ready", "ok");
  setProgress(0, "Idle");
}

document.addEventListener("DOMContentLoaded", init);
