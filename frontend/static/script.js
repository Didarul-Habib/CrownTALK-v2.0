// CrownTALK EXTREME v3 – Frontend
// - Up to 2 comments per tweet
// - Each comment has its own copy button
// - Comments based on tweet context, respecting multi-language logic (handled by backend)
// - Links inside comments clickable + shortened
// - Comment length clamped to 5–12 words
// - Global color skin (gold / obsidian / emerald / violet)

// ------------------ Backend config ------------------

const BACKEND_BASE = "https://crowntalk-v2-0.onrender.com";
const BACKEND_URL = `${BACKEND_BASE}/api/generate`;

/** @type {AbortController | null} */
let currentController = null;

/** Clipboard history in memory */
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
  colorThemeSwitcher: document.getElementById("ct-color-theme-switcher"),
};

// Color theme config
const COLOR_THEME_KEY = "crowntalk-color-theme";
const ALLOWED_COLOR_THEMES = ["gold", "obsidian", "emerald", "violet"];
const DEFAULT_COLOR_THEME = "gold";

// Frontend safety filters – avoid generic spammy talk
const BANNED_PHRASES = [
  "as an ai",
  "in this digital age",
  "slay",
  "yass",
  "bestie",
  "queen",
  "thoughts",
  "agree",
  "who's with me",
  "whos with me",
  "game changer",
  "transformative",
  "love this",
  "love that",
  "love it",
  "amazing",
  "awesome",
  "incredible",
  "finally",
  "excited",
];

// ------------------ Helper functions ------------------

function setStatus(status, type) {
  if (!els.statusPill) return;
  els.statusPill.textContent = status;
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

function getActiveThemeId() {
  if (!els.themeGrid) return "default";
  const btn = els.themeGrid.querySelector(".ct-theme-btn.is-active");
  return btn ? btn.dataset.themeId || "default" : "default";
}

// HTML escaping + link formatting
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function shortenUrlLabel(url) {
  if (!url) return "";
  const noProto = url.replace(/^https?:\/\//, "");
  if (noProto.length <= 42) return noProto;
  return noProto.slice(0, 39) + "…";
}

function displayUrlLabel(url) {
  if (!url) return "";
  return shortenUrlLabel(url);
}

function formatCommentHtml(text) {
  if (!text) return "";
  const escaped = escapeHtml(text);
  const withBreaks = escaped.replace(/\n/g, "<br>");
  const urlRegex = /(https?:\/\/[^\s<]+)/g;
  return withBreaks.replace(urlRegex, (match) => {
    const label = shortenUrlLabel(match);
    return `<a href="${match}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
}

// Remove banned phrases (case-insensitive) but keep rest of the text
function stripBanned(text) {
  if (!text) return "";
  let result = text;
  for (const phrase of BANNED_PHRASES) {
    const safe = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(safe, "gi");
    result = result.replace(re, "");
  }
  return result.replace(/\s+/g, " ").trim();
}

// Enforce word count limits
function clampWords(text, min = 5, max = 12) {
  if (!text) return "";
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return "";

  if (words.length > max) {
    return words.slice(0, max).join(" ");
  }

  if (words.length < min) {
    // pad by repeating last word so we never fall under min
    const lastWord = words[words.length - 1];
    while (words.length < min) {
      words.push(lastWord);
    }
  }

  return words.join(" ");
}

// Full preprocessing step for a single comment string
function preprocessComment(raw) {
  const stripped = stripBanned(raw || "");
  return clampWords(stripped, 5, 12);
}

function setSkeletonVisible(visible) {
  if (!els.skeletonContainer) return;
  els.skeletonContainer.classList.toggle("is-hidden", !visible);
}

function clearResults() {
  if (els.resultsList) els.resultsList.innerHTML = "";
}

// ------------------ Clipboard history ------------------

function pushClipboardHistory(entry) {
  const timestamp = new Date();
  clipboardHistory.unshift({ ...entry, time: timestamp });
  if (clipboardHistory.length > 40) clipboardHistory.pop();
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
    themeSpan.textContent = item.label || item.source || "Comment";

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

function copyText(text, meta) {
  if (!text) return;
  navigator.clipboard
    .writeText(text)
    .then(() => {
      pushClipboardHistory({ text, ...meta });
      setStatus("Copied to clipboard", "ok");
    })
    .catch(() => {
      setStatus("Copy failed (clipboard blocked)", "error");
    });
}

// ------------------ History drawer ------------------

function openHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.add("is-open");
}

function closeHistoryDrawer() {
  if (!els.historyDrawer) return;
  els.historyDrawer.classList.remove("is-open");
}

// ------------------ Color theme handling ------------------

function applyColorTheme(themeId) {
  if (!ALLOWED_COLOR_THEMES.includes(themeId)) {
    themeId = DEFAULT_COLOR_THEME;
  }

  document.body.setAttribute("data-color-theme", themeId);

  if (els.colorThemeSwitcher) {
    const dots = els.colorThemeSwitcher.querySelectorAll("[data-color-theme]");
    dots.forEach((btn) => {
      const active = btn.getAttribute("data-color-theme") === themeId;
      btn.classList.toggle("is-active", active);
    });
  }

  try {
    localStorage.setItem(COLOR_THEME_KEY, themeId);
  } catch {
    // ignore if storage not available
  }
}

// ------------------ Comment collection + rendering ------------------

// Supports multiple formats from backend:
// - item.comments = [ "c1", "c2" ]
// - item.comment = { en: "...", native: "...", ... }
// - item.comment = "single string"
function collectCommentsFromItem(item) {
  const result = [];

  if (Array.isArray(item.comments)) {
    for (const c of item.comments) {
      if (typeof c === "string") result.push(c);
    }
  }

  const cObj = item.comment;
  if (typeof cObj === "string") {
    result.push(cObj);
  } else if (cObj && typeof cObj === "object") {
    Object.values(cObj).forEach((val) => {
      if (typeof val === "string") result.push(val);
    });
  }

  const unique = [];
  const seen = new Set();
  for (const raw of result) {
    const trimmed = (raw || "").trim();
    if (!trimmed) continue;
    const key = trimmed.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(trimmed);
  }

  // Up to 2 comments per tweet
  return unique.slice(0, 2);
}

function renderCommentBlock(index, text, themeLabel, mood) {
  if (!text) return null;

  const block = document.createElement("div");
  block.className = "ct-comment-block";

  const labelRow = document.createElement("div");
  labelRow.style.display = "flex";
  labelRow.style.justifyContent = "space-between";
  labelRow.style.alignItems = "center";
  labelRow.style.gap = "6px";

  const labelEl = document.createElement("div");
  labelEl.className = "ct-comment-label";
  labelEl.textContent = `Comment ${index}`;

  const actions = document.createElement("div");
  actions.className = "ct-card-actions";

  const btnCopy = document.createElement("button");
  btnCopy.className = "ct-card-btn";
  btnCopy.textContent = "⧉ Copy";
  btnCopy.addEventListener("click", () => {
    copyText(text, {
      label: `Comment ${index}`,
      theme: themeLabel,
      mood,
    });
  });

  actions.appendChild(btnCopy);
  labelRow.appendChild(labelEl);
  labelRow.appendChild(actions);
  block.appendChild(labelRow);

  const p = document.createElement("p");
  p.className = "ct-comment-text";
  p.innerHTML = formatCommentHtml(text);
  block.appendChild(p);

  return block;
}

function fillCardFromItem(card, item) {
  card.innerHTML = "";

  const hasError = !!item.error;
  const themeLabel =
    (item.meta && (item.meta.theme_label || item.meta.theme_id)) ||
    getActiveThemeId();
  const mood = (item.meta && item.meta.mood) || "neutral";
  const keywords = (item.meta && item.meta.keywords) || [];
  const excerpt = (item.meta && item.meta.excerpt) || "";
  const authorName = (item.meta && item.meta.author_name) || "";

  // Header
  const header = document.createElement("div");
  header.className = "ct-card-header";

  const urlEl = document.createElement("a");
  urlEl.className = "ct-card-url";
  urlEl.href = item.url || "#";
  urlEl.target = "_blank";
  urlEl.rel = "noopener noreferrer";
  urlEl.textContent = displayUrlLabel(item.url || "");
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
    const rawComments = collectCommentsFromItem(item);

    if (!rawComments.length) {
      const p = document.createElement("p");
      p.className = "ct-comment-text ct-comment-text--faded";
      p.textContent = "No comment returned for this tweet.";
      body.appendChild(p);
    } else {
      rawComments.forEach((raw, idx) => {
        const processed = preprocessComment(raw);
        const block = renderCommentBlock(idx + 1, processed, themeLabel, mood);
        if (block) body.appendChild(block);
      });
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

  // Footer actions (reroll)
  const actions = document.createElement("div");
  actions.className = "ct-card-actions";

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
}

function renderResultItem(item) {
  if (!els.resultsList) return;
  const card = document.createElement("article");
  card.className = "ct-card";
  fillCardFromItem(card, item);
  els.resultsList.appendChild(card);
}

// ------------------ Reroll logic ------------------

async function rerollSingle(url, cardNode) {
  if (!url) return;
  const theme = getActiveThemeId();

  const controller = new AbortController();

  try {
    setStatus("Rerolling…", "ok");
    cardNode.style.opacity = "0.5";

    const payload = {
      urls: [url],
      theme,
    };

    const res = await fetch(BACKEND_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const batches = json.batches || [];
    if (!batches.length || !batches[0].items || !batches[0].items[0]) {
      throw new Error("Invalid reroll response");
    }
    const freshItem = batches[0].items[0];
    fillCardFromItem(cardNode, freshItem);
    setStatus("Reroll done", "ok");
  } catch (err) {
    console.error("Reroll failed", err);
    setStatus("Reroll failed", "error");
  } finally {
    cardNode.style.opacity = "1";
  }
}

// ------------------ Main generate flow ------------------

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

  clearResults();
  setSkeletonVisible(true);
  setProgress(8, "Preparing batch…");
  setStatus("Contacting backend…", "ok");

  if (els.btnGenerate) els.btnGenerate.disabled = true;
  if (els.btnCancel) els.btnCancel.disabled = false;
  if (els.btnClear) els.btnClear.disabled = true;

  const controller = new AbortController();
  currentController = controller;

  try {
    const payload = { urls, theme };

    const res = await fetch(BACKEND_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    setProgress(30, "Processing…");
    const json = await res.json();

    const batches = json.batches || [];
    const totalUrls = (json.meta && json.meta.total_urls) || urls.length || 1;
    let processed = 0;

    setSkeletonVisible(false);

    batches.forEach((batch) => {
      const items = batch.items || batch.results || [];
      items.forEach((item) => {
        processed += 1;
        const percent = 30 + Math.round((processed / totalUrls) * 70);
        setProgress(percent, `Rendering ${processed}/${totalUrls}…`);
        renderResultItem(item);
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
    if (els.btnGenerate) els.btnGenerate.disabled = false;
    if (els.btnCancel) els.btnCancel.disabled = true;
    if (els.btnClear) els.btnClear.disabled = false;
  }
}

// ------------------ Simple handlers ------------------

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
  const all = els.themeGrid.querySelectorAll(".ct-theme-btn");
  all.forEach((b) => b.classList.remove("is-active"));
  btn.classList.add("is-active");
}

// ------------------ Init ------------------

function init() {
  if (els.backendUrl) {
    els.backendUrl.textContent = BACKEND_BASE;
  }

  // Initialize color theme from storage
  let savedTheme = DEFAULT_COLOR_THEME;
  try {
    const stored = localStorage.getItem(COLOR_THEME_KEY);
    if (stored && ALLOWED_COLOR_THEMES.includes(stored)) {
      savedTheme = stored;
    }
  } catch {
    // ignore
  }
  applyColorTheme(savedTheme);

  if (els.colorThemeSwitcher) {
    els.colorThemeSwitcher.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-color-theme]");
      if (!btn) return;
      const themeId = btn.getAttribute("data-color-theme") || DEFAULT_COLOR_THEME;
      applyColorTheme(themeId);
    });
  }

  // Wire buttons
  if (els.btnGenerate) els.btnGenerate.addEventListener("click", handleGenerate);
  if (els.btnCancel) els.btnCancel.addEventListener("click", handleCancel);
  if (els.btnClear) els.btnClear.addEventListener("click", handleClear);
  if (els.themeGrid) els.themeGrid.addEventListener("click", handleThemeClick);

  // History drawer
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
