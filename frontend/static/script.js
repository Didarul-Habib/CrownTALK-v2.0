// CrownTALK EXTREME Frontend
// Wire to backend: https://crowntalk-v2-0.onrender.com

const BACKEND_URL = "https://crowntalk-v2-0.onrender.com/api/generate";

/** @type {AbortController | null} */
let currentController = null;

/** clipboard history in memory */
const clipboardHistory = [];

/** DOM refs */
const els = {
  backendUrl: document.getElementById("ct-backend-url"),
  urlsInput: document.getElementById("ct-urls-input"),
  themeGrid: document.getElementById("ct-theme-grid"),
  langOptions: document.getElementById("ct-lang-options"),
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

/** Util: set status pill */
function setStatus(status, type) {
  els.statusPill.textContent = status;
  els.statusPill.classList.remove("ct-status-pill--ok", "ct-status-pill--error");
  if (type === "ok") els.statusPill.classList.add("ct-status-pill--ok");
  if (type === "error") els.statusPill.classList.add("ct-status-pill--error");
}

/** Util: set progress */
function setProgress(percent, label) {
  els.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  if (label) {
    els.progressLabel.textContent = label;
  }
}

/** Util: parse URLs from textarea */
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

/** Get current theme id from UI */
function getActiveThemeId() {
  const btn = els.themeGrid.querySelector(".ct-theme-btn.is-active");
  return btn ? btn.dataset.themeId || "default" : "default";
}

/** Get language mode from UI */
function getLanguageMode() {
  const checked = els.langOptions.querySelector("input[name=language_mode]:checked");
  return checked ? checked.value : "dual";
}

/** Toggle skeleton loading */
function setSkeletonVisible(visible) {
  els.skeletonContainer.classList.toggle("is-hidden", !visible);
}

/** Clear results list */
function clearResults() {
  els.resultsList.innerHTML = "";
}

/** Render a single result card */
function renderResultItem(item, batchIndex, indexWithinBatch, total) {
  const card = document.createElement("article");
  card.className = "ct-card";

  const hasError = !!item.error;
  const themeLabel = item?.meta?.theme_label || item?.meta?.theme_id || "—";
  const mood = item?.meta?.mood || "neutral";
  const keywords = item?.meta?.keywords || [];
  const excerpt = item?.meta?.excerpt || "";
  const authorName = item?.meta?.author_name || "";

  const comment = item.comment || {};
  const enText = comment.en || "";
  const bnText = comment.bn || "";

  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("div");
  urlEl.className = "ct-card-url";
  urlEl.textContent = item.url || "";

  const chipRow = document.createElement("div");
  chipRow.className = "ct-card-chip-row";

  const chipTheme = document.createElement("span");
  chipTheme.className = "ct-chip ct-chip--theme";
  chipTheme.textContent = `Theme: ${themeLabel}`;
  chipRow.appendChild(chipTheme);

  const chipMood = document.createElement("span");
  chipMood.className = "ct-chip ct-chip--mood";
  chipMood.textContent = `Mood: ${mood}`;
  chipRow.appendChild(chipMood);

  if (hasError) {
    const chipErr = document.createElement("span");
    chipErr.className = "ct-chip";
    chipErr.textContent = `Error: ${item.error}`;
    chipRow.appendChild(chipErr);
  }

  header.appendChild(urlEl);
  header.appendChild(chipRow);

  const body = document.createElement("div");
  body.className = "ct-card-body";

  if (hasError) {
    const p = document.createElement("p");
    p.className = "ct-comment-text ct-comment-text--faded";
    p.textContent = item.message || "Something went wrong for this URL.";
    body.appendChild(p);
  } else {
    if (enText) {
      const blockEn = document.createElement("div");
      blockEn.className = "ct-comment-block";
      const labelEn = document.createElement("div");
      labelEn.className = "ct-comment-label";
      labelEn.textContent = "English";
      const pEn = document.createElement("p");
      pEn.className = "ct-comment-text";
      pEn.textContent = enText;
      blockEn.appendChild(labelEn);
      blockEn.appendChild(pEn);
      body.appendChild(blockEn);
    }

    if (bnText) {
      const blockBn = document.createElement("div");
      blockBn.className = "ct-comment-block";
      const labelBn = document.createElement("div");
      labelBn.className = "ct-comment-label";
      labelBn.textContent = "Bangla";
      const pBn = document.createElement("p");
      pBn.className = "ct-comment-text ct-comment-text--faded";
      pBn.textContent = bnText;
      blockBn.appendChild(labelBn);
      blockBn.appendChild(pBn);
      body.appendChild(blockBn);
    }
  }

  const metaFooter = document.createElement("div");
  metaFooter.className = "ct-meta-footer";

  const keywordsText = keywords.length ? keywords.join(", ") : "—";
  const metaLeft = document.createElement("span");
  metaLeft.className = "ct-meta-keywords";
  metaLeft.textContent = `Keywords: ${keywordsText}`;

  const metaRight = document.createElement("span");
  metaRight.textContent =
    (authorName ? `@${authorName}` : "unknown") +
    (excerpt ? ` · “${excerpt}”` : "");

  metaFooter.appendChild(metaLeft);
  metaFooter.appendChild(metaRight);

  const actions = document.createElement("div");
  actions.className = "ct-card-actions";

  const btnCopy = document.createElement("button");
  btnCopy.className = "ct-card-btn";
  btnCopy.innerHTML = "⧉ Copy";
  btnCopy.addEventListener("click", () => {
    const langMode = getLanguageMode();
    let textToCopy = "";
    if (langMode === "en") textToCopy = enText || bnText;
    else if (langMode === "bn") textToCopy = bnText || enText;
    else {
      textToCopy = [enText, bnText].filter(Boolean).join("\n\n");
    }
    if (!textToCopy) return;

    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        pushClipboardHistory({
          theme: themeLabel,
          mood,
          text: textToCopy,
        });
        setStatus("Copied to clipboard", "ok");
      })
      .catch(() => {
        setStatus("Copy failed (clipboard blocked)", "error");
      });
  });

  const btnReroll = document.createElement("button");
  btnReroll.className = "ct-card-btn";
  btnReroll.innerHTML = "🔁 Reroll";
  btnReroll.addEventListener("click", () => {
    rerollSingle(item.url, card);
  });

  actions.appendChild(btnCopy);
  if (!hasError) {
    actions.appendChild(btnReroll);
  }

  card.appendChild(header);
  card.appendChild(body);
  card.appendChild(metaFooter);
  card.appendChild(actions);

  els.resultsList.appendChild(card);
}

/** Reroll a single URL */
async function rerollSingle(url, cardNode) {
  if (!url) return;
  const theme = getActiveThemeId();
  const language = getLanguageMode();

  const localController = new AbortController();

  try {
    setStatus("Rerolling…", "ok");

    cardNode.style.opacity = "0.5";

    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: localController.signal,
      body: JSON.stringify({
        urls: [url],
        theme,
        language_mode: language,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const json = await res.json();
    const batches = json.batches || [];
    if (!batches.length || !batches[0].items || !batches[0].items[0]) {
      throw new Error("Invalid reroll response");
    }

    const freshItem = batches[0].items[0];

    // Remove old content and render new inside same card
    cardNode.innerHTML = "";
    renderResultItemIntoExisting(freshItem, cardNode);
    setStatus("Reroll done", "ok");
  } catch (err) {
    console.error("Reroll failed", err);
    setStatus("Reroll failed", "error");
  } finally {
    cardNode.style.opacity = "1";
  }
}

/** Reuse same render logic but into an existing card node */
function renderResultItemIntoExisting(item, card) {
  // Minimal version of renderResultItem, reusing logic:

  const hasError = !!item.error;
  const themeLabel = item?.meta?.theme_label || item?.meta?.theme_id || "—";
  const mood = item?.meta?.mood || "neutral";
  const keywords = item?.meta?.keywords || [];
  const excerpt = item?.meta?.excerpt || "";
  const authorName = item?.meta?.author_name || "";

  const comment = item.comment || {};
  const enText = comment.en || "";
  const bnText = comment.bn || "";

  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("div");
  urlEl.className = "ct-card-url";
  urlEl.textContent = item.url || "";

  const chipRow = document.createElement("div");
  chipRow.className = "ct-card-chip-row";

  const chipTheme = document.createElement("span");
  chipTheme.className = "ct-chip ct-chip--theme";
  chipTheme.textContent = `Theme: ${themeLabel}`;
  chipRow.appendChild(chipTheme);

  const chipMood = document.createElement("span");
  chipMood.className = "ct-chip ct-chip--mood";
  chipMood.textContent = `Mood: ${mood}`;
  chipRow.appendChild(chipMood);

  if (hasError) {
    const chipErr = document.createElement("span");
    chipErr.className = "ct-chip";
    chipErr.textContent = `Error: ${item.error}`;
    chipRow.appendChild(chipErr);
  }

  header.appendChild(urlEl);
  header.appendChild(chipRow);

  const body = document.createElement("div");
  body.className = "ct-card-body";

  if (hasError) {
    const p = document.createElement("p");
    p.className = "ct-comment-text ct-comment-text--faded";
    p.textContent = item.message || "Something went wrong for this URL.";
    body.appendChild(p);
  } else {
    if (enText) {
      const blockEn = document.createElement("div");
      blockEn.className = "ct-comment-block";
      const labelEn = document.createElement("div");
      labelEn.className = "ct-comment-label";
      labelEn.textContent = "English";
      const pEn = document.createElement("p");
      pEn.className = "ct-comment-text";
      pEn.textContent = enText;
      blockEn.appendChild(labelEn);
      blockEn.appendChild(pEn);
      body.appendChild(blockEn);
    }

    if (bnText) {
      const blockBn = document.createElement("div");
      blockBn.className = "ct-comment-block";
      const labelBn = document.createElement("div");
      labelBn.className = "ct-comment-label";
      labelBn.textContent = "Bangla";
      const pBn = document.createElement("p");
      pBn.className = "ct-comment-text ct-comment-text--faded";
      pBn.textContent = bnText;
      blockBn.appendChild(labelBn);
      blockBn.appendChild(pBn);
      body.appendChild(blockBn);
    }
  }

  const metaFooter = document.createElement("div");
  metaFooter.className = "ct-meta-footer";

  const keywordsText = keywords.length ? keywords.join(", ") : "—";
  const metaLeft = document.createElement("span");
  metaLeft.className = "ct-meta-keywords";
  metaLeft.textContent = `Keywords: ${keywordsText}`;

  const metaRight = document.createElement("span");
  metaRight.textContent =
    (authorName ? `@${authorName}` : "unknown") +
    (excerpt ? ` · “${excerpt}”` : "");

  metaFooter.appendChild(metaLeft);
  metaFooter.appendChild(metaRight);

  const actions = document.createElement("div");
  actions.className = "ct-card-actions";

  const btnCopy = document.createElement("button");
  btnCopy.className = "ct-card-btn";
  btnCopy.innerHTML = "⧉ Copy";
  btnCopy.addEventListener("click", () => {
    const langMode = getLanguageMode();
    let textToCopy = "";
    if (langMode === "en") textToCopy = enText || bnText;
    else if (langMode === "bn") textToCopy = bnText || enText;
    else {
      textToCopy = [enText, bnText].filter(Boolean).join("\n\n");
    }
    if (!textToCopy) return;

    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        pushClipboardHistory({
          theme: themeLabel,
          mood,
          text: textToCopy,
        });
        setStatus("Copied to clipboard", "ok");
      })
      .catch(() => {
        setStatus("Copy failed (clipboard blocked)", "error");
      });
  });

  const btnReroll = document.createElement("button");
  btnReroll.className = "ct-card-btn";
  btnReroll.innerHTML = "🔁 Reroll";
  btnReroll.addEventListener("click", () => {
    rerollSingle(item.url, card);
  });

  actions.appendChild(btnCopy);
  if (!hasError) {
    actions.appendChild(btnReroll);
  }

  card.appendChild(header);
  card.appendChild(body);
  card.appendChild(metaFooter);
  card.appendChild(actions);
}

/** Clipboard history handling */
function pushClipboardHistory(entry) {
  const timestamp = new Date();
  clipboardHistory.unshift({
    ...entry,
    time: timestamp,
  });
  if (clipboardHistory.length > 20) clipboardHistory.pop();
  renderClipboardHistory();
}

function renderClipboardHistory() {
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
    themeSpan.textContent = `${item.theme} · ${item.mood}`;

    const timeSpan = document.createElement("span");
    timeSpan.className = "ct-history-item-time";
    timeSpan.textContent = item.time.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    header.appendChild(themeSpan);
    header.appendChild(timeSpan);

    const body = document.createElement("div");
    body.className = "ct-history-item-body";
    body.textContent = item.text;

    wrap.appendChild(header);
    wrap.appendChild(body);
    container.appendChild(wrap);
  });
}

/** History drawer toggling */
function openHistoryDrawer() {
  els.historyDrawer.classList.add("is-open");
}

function closeHistoryDrawer() {
  els.historyDrawer.classList.remove("is-open");
}

/** Main run handler */
async function handleGenerate() {
  const rawUrls = els.urlsInput.value;
  const urls = parseUrls(rawUrls);

  if (!urls.length) {
    setStatus("Paste at least one URL", "error");
    return;
  }

  if (currentController) {
    // Safety: don’t allow concurrent
    setStatus("Already running, cancel first", "error");
    return;
  }

  const theme = getActiveThemeId();
  const language = getLanguageMode();

  clearResults();
  setSkeletonVisible(true);
  setProgress(8, "Preparing batch…");
  setStatus("Contacting backend…", "ok");

  els.btnGenerate.disabled = true;
  els.btnCancel.disabled = false;
  els.btnClear.disabled = true;

  const controller = new AbortController();
  currentController = controller;

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        urls,
        theme,
        language_mode: language,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    setProgress(30, "Processing…");

    const json = await res.json();

    const batches = json.batches || [];
    const totalUrls = json.meta?.total_urls || urls.length || 1;
    let processed = 0;

    setSkeletonVisible(false);

    batches.forEach((batch, idx) => {
      const items = batch.items || [];
      items.forEach((item, itemIdx) => {
        processed += 1;
        const percent = 30 + Math.round((processed / totalUrls) * 70);
        setProgress(percent, `Rendering ${processed}/${totalUrls}…`);
        renderResultItem(item, idx, itemIdx, totalUrls);
      });
    });

    setStatus("Done", "ok");
    setProgress(100, "Completed");
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Run cancelled", "error");
      setSkeletonVisible(false);
      setProgress(0, "Cancelled");
    } else {
      console.error("Generate failed", err);
      setStatus("Backend error, try again", "error");
      setSkeletonVisible(false);
      setProgress(0, "Error");
    }
  } finally {
    currentController = null;
    els.btnGenerate.disabled = false;
    els.btnCancel.disabled = true;
    els.btnClear.disabled = false;
  }
}

/** Cancel handler */
function handleCancel() {
  if (currentController) {
    currentController.abort();
  }
}

/** Clear input */
function handleClear() {
  els.urlsInput.value = "";
  setStatus("Input cleared", "ok");
}

/** Theme click handler */
function handleThemeClick(event) {
  const btn = event.target.closest(".ct-theme-btn");
  if (!btn) return;
  els.themeGrid
    .querySelectorAll(".ct-theme-btn")
    .forEach((b) => b.classList.remove("is-active"));
  btn.classList.add("is-active");
}

/** Wire up events */
function init() {
  if (els.backendUrl) {
    els.backendUrl.textContent = BACKEND_URL.replace("/api/generate", "");
  }

  els.btnGenerate.addEventListener("click", handleGenerate);
  els.btnCancel.addEventListener("click", handleCancel);
  els.btnClear.addEventListener("click", handleClear);
  els.themeGrid.addEventListener("click", handleThemeClick);

  if (els.historyToggleBtn) {
    els.historyToggleBtn.addEventListener("click", () => {
      if (els.historyDrawer.classList.contains("is-open")) {
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
      clipboardHistory.length = 0;
      renderClipboardHistory();
    });
  }

  // ESC closes history on desktop
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeHistoryDrawer();
  });

  renderClipboardHistory();
}

document.addEventListener("DOMContentLoaded", init);
