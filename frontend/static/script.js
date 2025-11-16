// CrownTALK EXTREME v3 – generator-based frontend
// Backend base URL
const BACKEND_BASE = "https://crowntalk-v2-0.onrender.com";
const BACKEND_URL = `${BACKEND_BASE}/api/generate`;

/** @type {AbortController | null} */
let currentController = null;

/** clipboard history in memory */
const clipboardHistory = [];

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

function setStatus(status, type) {
  if (!els.statusPill) return;
  els.statusPill.textContent = status;
  els.statusPill.classList.remove("ct-status-pill--ok", "ct-status-pill--error");
  if (type === "ok") els.statusPill.classList.add("ct-status-pill--ok");
  if (type === "error") els.statusPill.classList.add("ct-status-pill--error");
}

function setProgress(percent, label) {
  if (els.progressFill) {
    els.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }
  if (label && els.progressLabel) {
    els.progressLabel.textContent = label;
  }
}

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

/** Theme id from UI (for label only) */
function getActiveThemeId() {
  if (!els.themeGrid) return "default";
  const btn = els.themeGrid.querySelector(".ct-theme-btn.is-active");
  return btn ? btn.dataset.themeId || "default" : "default";
}

/** We always ask backend for dual language now */
function getLanguageMode() {
  return "dual";
}

/** Toggle skeleton loading */
function setSkeletonVisible(visible) {
  if (!els.skeletonContainer) return;
  els.skeletonContainer.classList.toggle("is-hidden", !visible);
}

/** Clear results list */
function clearResults() {
  if (els.resultsList) els.resultsList.innerHTML = "";
}

/* ------------------ Clipboard history ------------------ */

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
    themeSpan.textContent = item.label || `${item.theme || "Comment"} · ${item.mood || ""}`;

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

function copyText(text, meta = {}) {
  if (!text) return;
  navigator.clipboard
    .writeText(text)
    .then(() => {
      pushClipboardHistory({
        ...meta,
        text,
      });
      setStatus("Copied to clipboard", "ok");
    })
    .catch(() => {
      setStatus("Copy failed (clipboard blocked)", "error");
    });
}

/* ------------------ History drawer ------------------ */

function openHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.add("is-open");
}

function closeHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.remove("is-open");
}

/* ------------------ Card rendering ------------------ */

function renderResultItem(item, batchIndex, indexWithinBatch, totalCount) {
  if (!els.resultsList) return;

  const card = document.createElement("article");
  card.className = "ct-card";

  const hasError = !!item.error;
  const themeLabel = item?.meta?.theme_label || item?.meta?.theme_id || getActiveThemeId();
  const mood = item?.meta?.mood || "neutral";
  const keywords = item?.meta?.keywords || [];
  const excerpt = item?.meta?.excerpt || "";
  const authorName = item?.meta?.author_name || "";

  const comment = item.comment || {};
  const enText = comment.en || "";
  const bnText = comment.bn || "";

  // Header
  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("a");
  urlEl.className = "ct-card-url";
  urlEl.href = item.url || "#";
  urlEl.target = "_blank";
  urlEl.rel = "noopener noreferrer";
  urlEl.textContent = item.url || "";
  header.appendChild(urlEl);

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

  header.appendChild(chipRow);

  // Body
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
      labelEn.textContent = "ENGLISH";
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
      labelBn.textContent = "BANGLA";
      const pBn = document.createElement("p");
      pBn.className = "ct-comment-text ct-comment-text--faded";
      pBn.textContent = bnText;
      blockBn.appendChild(labelBn);
      blockBn.appendChild(pBn);
      body.appendChild(blockBn);
    }
  }

  // Meta footer
  const metaFooter = document.createElement("div");
  metaFooter.className = "ct-meta-footer";

  const keywordsText = keywords.length ? keywords.join(", ") : "—";
  const metaLeft = document.createElement("span");
  metaLeft.className = "ct-meta-keywords";
  metaLeft.textContent = `Keywords: ${keywordsText}`;

  const metaRight = document.createElement("span");
  const shortExcerpt =
    excerpt && excerpt.length > 120 ? `${excerpt.slice(0, 117)}…` : excerpt;
  metaRight.textContent =
    (authorName ? `@${authorName}` : "unknown") +
    (shortExcerpt ? ` · “${shortExcerpt}”` : "");

  metaFooter.appendChild(metaLeft);
  metaFooter.appendChild(metaRight);

  // Card actions – per-language copy + reroll
  const actions = document.createElement("div");
  actions.className = "ct-card-actions";

  if (!hasError) {
    if (enText) {
      const btnCopyEn = document.createElement("button");
      btnCopyEn.className = "ct-card-btn";
      btnCopyEn.textContent = "⧉ Copy EN";
      btnCopyEn.addEventListener("click", () => {
        copyText(enText, { theme: themeLabel, mood, label: "English" });
      });
      actions.appendChild(btnCopyEn);
    }

    if (bnText) {
      const btnCopyBn = document.createElement("button");
      btnCopyBn.className = "ct-card-btn";
      btnCopyBn.textContent = "⧉ Copy Bangla";
      btnCopyBn.addEventListener("click", () => {
        copyText(bnText, { theme: themeLabel, mood, label: "Bangla" });
      });
      actions.appendChild(btnCopyBn);
    }

    if (enText || bnText) {
      const btnCopyBoth = document.createElement("button");
      btnCopyBoth.className = "ct-card-btn";
      btnCopyBoth.textContent = "⧉ Copy both";
      btnCopyBoth.addEventListener("click", () => {
        const combined = [enText, bnText].filter(Boolean).join("\n\n");
        copyText(combined, { theme: themeLabel, mood, label: "EN+BN" });
      });
      actions.appendChild(btnCopyBoth);
    }
  }

  const btnReroll = document.createElement("button");
  btnReroll.className = "ct-card-btn";
  btnReroll.textContent = "🔁 Reroll";
  btnReroll.addEventListener("click", () => {
    rerollSingle(item.url, card);
  });
  actions.appendChild(btnReroll);

  card.appendChild(header);
  card.appendChild(body);
  card.appendChild(metaFooter);
  card.appendChild(actions);

  els.resultsList.appendChild(card);
}

/* ------------------ Reroll ------------------ */

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

    // Rebuild card in-place
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

function renderResultItemIntoExisting(item, card) {
  // reuse main render with slight adaptation
  card.innerHTML = "";
  const tmpList = document.createElement("div");
  els.resultsList.appendChild(tmpList); // just to reuse function? nope

  // hack: temporarily point resultsList to the card's parent
  const originalList = els.resultsList;
  els.resultsList = card.parentElement || originalList;
  renderResultItem(item, 0, 0, 1);
  els.resultsList = originalList;
  // remove the new card we just appended & move its content into 'card'
  const newCard = (card.parentElement || originalList).lastElementChild;
  if (newCard && newCard !== card) {
    card.innerHTML = newCard.innerHTML;
    newCard.remove();
  }
}

/* ------------------ Generate main flow ------------------ */

async function handleGenerate() {
  const rawUrls = els.urlsInput ? els.urlsInput.value : "";
  const urls = parseUrls(rawUrls);

  if (!urls.length) {
    setStatus("Paste at least one URL", "error");
    return;
  }

  if (currentController) {
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
        language_mode: language, // always "dual"
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

/* ------------------ Other handlers ------------------ */

function handleCancel() {
  if (currentController) {
    currentController.abort();
  }
}

function handleClear() {
  if (els.urlsInput) els.urlsInput.value = "";
  setStatus("Input cleared", "ok");
}

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
  // show backend base (without /api/generate)
  if (els.backendUrl) {
    els.backendUrl.textContent = BACKEND_BASE;
  }

  // hide language mode block visually (we force dual under the hood)
  const langOptions = document.getElementById("ct-lang-options");
  if (langOptions) {
    const block = langOptions.closest(".ct-section-block");
    if (block) block.style.display = "none";
  }

  if (els.btnGenerate) els.btnGenerate.addEventListener("click", handleGenerate);
  if (els.btnCancel) els.btnCancel.addEventListener("click", handleCancel);
  if (els.btnClear) els.btnClear.addEventListener("click", handleClear);
  if (els.themeGrid) els.themeGrid.addEventListener("click", handleThemeClick);

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
      clipboardHistory.length = 0;
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
