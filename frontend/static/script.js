/* ============================================
   CrownTALK — One-time Access Gate + App Logic
   Access code: @CrownTALK@2026@CrownDEX
   Persists with localStorage + cookie fallback
   ============================================ */

/* ---------- Gate (single source of truth) ---------- */
(() => {
  const ACCESS_CODE = '@CrownTALK@2026@CrownDEX';
  const STORAGE_KEY = 'crowntalk_access_v1';    // local/session storage key
  const COOKIE_KEY  = 'crowntalk_access_v1';    // cookie fallback

  function isAuthorized() {
    try { if (localStorage.getItem(STORAGE_KEY) === '1') return true; } catch {}
    try { if (sessionStorage.getItem(STORAGE_KEY) === '1') return true; } catch {}
    try { if (document.cookie.includes(`${COOKIE_KEY}=1`)) return true; } catch {}
    return false;
  }

  function markAuthorized() {
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch {}
    try { sessionStorage.setItem(STORAGE_KEY, '1'); } catch {}
    try {
      document.cookie = `${COOKIE_KEY}=1; max-age=${365*24*3600}; path=/; samesite=lax`;
    } catch {}
  }

  function els() {
    return {
      gate:  document.getElementById('adminGate'),
      input: document.getElementById('password'),
    };
  }

  function showGate() {
    const { gate, input } = els();
    if (!gate) return;
    gate.hidden = false;
    gate.style.display = 'grid';   // ensure visible even if other CSS overrides
    document.body.style.overflow = 'hidden';
    if (input) {
      input.value = '';
      setTimeout(() => input.focus(), 0);
    }
  }

  function hideGate() {
    const { gate } = els();
    if (!gate) return;
    gate.hidden = true;
    gate.style.display = 'none';
    document.body.style.overflow = '';
  }

  async function tryAuth() {
    const { input } = els();
    if (!input) return;
    const val = (input.value || '').trim();
    if (!val) return;

    if (val === ACCESS_CODE) {
      markAuthorized();
      hideGate();
      // boot app UI now that we’re unlocked
      bootAppUI();
      return;
    }

    // wrong code → subtle shake + hint
    input.classList.add('ct-shake');
    setTimeout(() => input.classList.remove('ct-shake'), 350);
    input.value = '';
    input.placeholder = 'Wrong code — try again';
  }

  function bindGate() {
    const { gate, input } = els();
    if (!gate) return;

    // Enter to submit
    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        tryAuth();
      }
    });

    // Click the lock icon to submit
    const lockIcon = gate.querySelector('svg');
    if (lockIcon) {
      lockIcon.style.cursor = 'pointer';
      lockIcon.addEventListener('click', tryAuth);
    }
  }

  function init() {
    if (isAuthorized()) {
      hideGate();
      bootAppUI();
    } else {
      showGate();
      bindGate();
    }
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ========= App code (unchanged, just wrapped) ========= */

// ------------------------
// Backend endpoints
// ------------------------
const backendBase = "https://crowntalk.onrender.com";
const commentURL  = `${backendBase}/comment`;
const rerollURL   = `${backendBase}/reroll`;

// ------------------------
// DOM elements
// ------------------------
const urlInput        = document.getElementById("urlInput");
const generateBtn     = document.getElementById("generateBtn");
const cancelBtn       = document.getElementById("cancelBtn");
const clearBtn        = document.getElementById("clearBtn");
const progressEl      = document.getElementById("progress");
const progressBarFill = document.getElementById("progressBarFill");
const resultsEl       = document.getElementById("results");
const failedEl        = document.getElementById("failed");
const resultCountEl   = document.getElementById("resultCount");
const failedCountEl   = document.getElementById("failedCount");
const historyEl       = document.getElementById("history");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const yearEl          = document.getElementById("year");

// theme dots (live node list -> array)
let themeDots = Array.from(document.querySelectorAll(".theme-dot"));

// ------------------------
// State
// ------------------------
let cancelled    = false;
let historyItems = [];

// ------------------------
// Backend helpers (warmup + timeout fetch)
// ------------------------
function warmBackendOnce() {
  // fire-and-forget ping to wake the Render instance
  try {
    fetch(backendBase + "/", {
      method: "GET",
      cache: "no-store",
      mode: "no-cors",
      keepalive: true,
    }).catch(() => {});
  } catch (err) {
    console.warn("warmBackendOnce error", err);
  }
}

// Only warm occasionally while user is actually active
let lastWarmAt = 0;

function maybeWarmBackend() {
  const now = Date.now();
  const FIVE_MIN = 5 * 60 * 1000;

  // Only ping if 5+ minutes since last warm
  if (now - lastWarmAt > FIVE_MIN) {
    lastWarmAt = now;
    warmBackendOnce();
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 45000) {
  // Older browsers: no AbortController → just use plain fetch
  if (typeof AbortController === "undefined") {
    return fetch(url, options);
  }

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return res;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

// ------------------------
// Utilities
// ------------------------
function parseURLs(raw) {
  if (!raw) return [];
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^\s*\d+\.\s*/, "").trim());
}

function setProgressText(text) {
  if (progressEl) progressEl.textContent = text || "";
}

function setProgressRatio(ratio) {
  if (!progressBarFill) return;
  const clamped = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
  progressBarFill.style.transform = `scaleX(${clamped})`;
}

function resetProgress() {
  setProgressText("");
  setProgressRatio(0);
}

function resetResults() {
  if (resultsEl) resultsEl.innerHTML = "";
  if (failedEl) failedEl.innerHTML = "";
  if (resultCountEl) resultCountEl.textContent = "0 tweets";
  if (failedCountEl) failedCountEl.textContent = "0";
}

async function copyToClipboard(text) {
  if (!text) return;

  // Modern async clipboard (no scroll / focus)
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (err) {
      console.warn("navigator.clipboard failed, using fallback", err);
    }
  }

  // Fallback: hidden span + Range, NO focus, NO scroll jump
  const helper = document.createElement("span");
  helper.textContent = text;
  helper.style.position = "fixed";
  helper.style.left = "-9999px";
  helper.style.top = "0";
  helper.style.whiteSpace = "pre"; // preserve line breaks

  document.body.appendChild(helper);

  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(helper);

  selection.removeAllRanges();
  selection.addRange(range);

  try {
    document.execCommand("copy");
  } catch (err) {
    console.error("execCommand copy failed", err);
  }

  selection.removeAllRanges();
  document.body.removeChild(helper);
}

function formatTweetCount(count) {
  const n = Number(count) || 0;
  return `${n} tweet${n === 1 ? "" : "s"}`;
}

function autoResizeTextarea() {
  if (!urlInput) return;
  urlInput.style.height = "auto";
  const base = 180;
  const newHeight = Math.max(base, urlInput.scrollHeight);
  urlInput.style.height = newHeight + "px";
}

// ------------------------
// History
// ------------------------
function addToHistory(text) {
  if (!text) return;
  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  historyItems.push({ text, timestamp });
  renderHistory();
}

function renderHistory() {
  if (!historyEl) return;
  historyEl.innerHTML = "";
  if (!historyItems.length) {
    historyEl.textContent = "Copied comments will show up here.";
    return;
  }
  [...historyItems].reverse().forEach((item) => {
    const entry = document.createElement("div");
    entry.className = "history-item";

    const textSpan = document.createElement("div");
    textSpan.className = "history-text";
    textSpan.textContent = item.text;

    const right = document.createElement("div");
    right.style.display = "flex";
    right.style.flexDirection = "column";
    right.style.alignItems = "flex-end";
    right.style.gap = "4px";

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = item.timestamp;

    const btn = document.createElement("button");
    btn.className = "history-copy-btn";

    const main = document.createElement("span");
    main.innerHTML = `
      <svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg"
           shape-rendering="geometricPrecision" text-rendering="geometricPrecision"
           image-rendering="optimizeQuality" fill-rule="evenodd" clip-rule="evenodd"
           viewBox="0 0 467 512.22">
        <path fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z">
        </path>
      </svg>
      Copy
    `;
    const alt = document.createElement("span");
    alt.textContent = "Copied";

    btn.appendChild(main);
    btn.appendChild(alt);
    btn.dataset.text = item.text;

    right.appendChild(meta);
    right.appendChild(btn);

    entry.appendChild(textSpan);
    entry.appendChild(right);

    historyEl.appendChild(entry);
  });
}

// ------------------------
// Rendering helpers
// ------------------------
function buildTweetBlock(result) {
  const url = result.url || "";
  const comments = Array.isArray(result.comments) ? result.comments : [];

  const tweet = document.createElement("div");
  tweet.className = "tweet";
  tweet.dataset.url = url;

  const header = document.createElement("div");
  header.className = "tweet-header";

  const link = document.createElement("a");
  link.className = "tweet-link";
  link.href = url || "#";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url || "(no url)";
  link.title = "Open tweet (tap to open)";

  const actions = document.createElement("div");
  actions.className = "tweet-actions";

  const rerollBtn = document.createElement("button");
  rerollBtn.className = "reroll-btn";
  rerollBtn.textContent = "Reroll";

  actions.appendChild(rerollBtn);
  header.appendChild(link);
  header.appendChild(actions);
  tweet.appendChild(header);

  const commentsWrap = document.createElement("div");
  commentsWrap.className = "comments";

  const hasNative = comments.some((c) => c && c.lang && c.lang !== "en");
  const multilingual = hasNative;

  comments.forEach((comment, idx) => {
    if (!comment || !comment.text) return;

    const line = document.createElement("div");
    line.className = "comment-line";
    if (comment.lang) line.dataset.lang = comment.lang;

    const tag = document.createElement("span");
    tag.className = "comment-tag";
    tag.textContent = multilingual
      ? (comment.lang === "en" ? "EN" : (comment.lang || "native").toUpperCase())
      : `EN #${idx + 1}`;

    const bubble = document.createElement("span");
    bubble.className = "comment-text";
    bubble.textContent = comment.text;

    const copyBtn = document.createElement("button");
    let copyLabel = "Copy";
    if (multilingual) {
      if (comment.lang === "en") {
        copyBtn.className = "copy-btn-en";
        copyLabel = "Copy EN";
      } else {
        copyBtn.className = "copy-btn";
      }
    } else {
      copyBtn.className = "copy-btn";
    }

    const copyMain = document.createElement("span");
    copyMain.innerHTML = `
      <svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg"
           shape-rendering="geometricPrecision" text-rendering="geometricPrecision"
           image-rendering="optimizeQuality" fill-rule="evenodd" clip-rule="evenodd"
           viewBox="0 0 467 512.22">
        <path fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z">
        </path>
      </svg>
      ${copyLabel}
    `;

    const copyAlt = document.createElement("span");
    copyAlt.textContent = "Copied";

    copyBtn.appendChild(copyMain);
    copyBtn.appendChild(copyAlt);
    copyBtn.dataset.text = comment.text;

    line.appendChild(tag);
    line.appendChild(bubble);
    line.appendChild(copyBtn);

    commentsWrap.appendChild(line);
  });

  tweet.appendChild(commentsWrap);
  return tweet;
}

function appendResultBlock(result) {
  const block = buildTweetBlock(result);
  resultsEl.appendChild(block);
}

function updateTweetBlock(tweetEl, result) {
  if (!tweetEl) return;
  tweetEl.dataset.url = result.url || tweetEl.dataset.url || "";

  const oldComments = tweetEl.querySelector(".comments");
  if (oldComments) oldComments.remove();

  const newComments = buildTweetBlock(result).querySelector(".comments");
  if (newComments) tweetEl.appendChild(newComments);

  tweetEl.style.transition = "box-shadow 0.25s ease, transform 0.25s ease";
  const oldShadow = tweetEl.style.boxShadow;
  const oldTransform = tweetEl.style.transform;
  tweetEl.style.boxShadow = "0 0 0 2px rgba(56,189,248,0.9)";
  tweetEl.style.transform = "translateY(-1px)";
  setTimeout(() => {
    tweetEl.style.boxShadow = oldShadow;
    tweetEl.style.transform = oldTransform;
  }, 420);
}

function appendFailedItem(failure) {
  const wrapper = document.createElement("div");
  wrapper.className = "failed-item";

  const urlSpan = document.createElement("div");
  urlSpan.className = "failed-url";
  urlSpan.textContent = failure.url || "(unknown URL)";

  const reasonSpan = document.createElement("div");
  reasonSpan.className = "failed-reason";
  reasonSpan.textContent = failure.reason || "Unknown error";

  wrapper.appendChild(urlSpan);
  wrapper.appendChild(reasonSpan);
  failedEl.appendChild(wrapper);
}

function showSkeletons(count) {
  resultsEl.innerHTML = "";
  const num = Math.min(Math.max(count, 1), 6);
  for (let i = 0; i < num; i++) {
    const sk = document.createElement("div");
    sk.className = "tweet-skeleton";
    for (let j = 0; j < 3; j++) {
      const line = document.createElement("div");
      line.className = "tweet-skeleton-line";
      sk.appendChild(line);
    }
    resultsEl.appendChild(sk);
  }
}

// ------------------------
// Generate flow
// ------------------------
async function handleGenerate() {
  const raw = urlInput.value;
  const urls = parseURLs(raw);

  if (!urls.length) {
    alert("Please paste at least one tweet URL.");
    return;
  }

  // Try to wake the backend if it's been idle for a while
  maybeWarmBackend();

  cancelled = false;
  document.body.classList.add("is-generating");

  generateBtn.disabled = true;
  cancelBtn.disabled   = false;
  resetResults();
  resetProgress();

  setProgressText(`Processing ${urls.length} URL${urls.length === 1 ? "" : "s"}…`);
  setProgressRatio(0.03);
  showSkeletons(urls.length);

  // Shared request body/options
  const payload = JSON.stringify({ urls });
  const requestOptions = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
  };

  try {
    let res;

    // ---- first attempt ----
    try {
      res = await fetchWithTimeout(commentURL, requestOptions, 45000);
    } catch (firstErr) {
      console.warn("First generate attempt failed, warming backend then retrying…", firstErr);
      setProgressText("Waking CrownTALK engine… retrying once.");
      warmBackendOnce();

      // ---- second (final) attempt ----
      res = await fetchWithTimeout(commentURL, requestOptions, 45000);
    }

    if (!res.ok) {
      throw new Error(`Backend error: ${res.status}`);
    }

    const data = await res.json();
    if (cancelled) return;

    const results = Array.isArray(data.results) ? data.results : [];
    const failed  = Array.isArray(data.failed)  ? data.failed  : [];

    resultsEl.innerHTML = "";
    failedEl.innerHTML  = "";

    let processed = 0;
    const total = results.length || urls.length;

    let delay = 50;
    results.forEach((item) => {
      setTimeout(() => {
        if (cancelled) return;

        appendResultBlock(item);
        processed += 1;

        const ratio = total ? processed / total : 1;
        setProgressRatio(ratio);
        setProgressText(`Processed ${processed}/${total} tweets…`);
        resultCountEl.textContent = formatTweetCount(processed);

        if (processed === total) {
          setProgressText(`Processed ${processed} tweet${processed === 1 ? "" : "s"}.`);
          document.body.classList.remove("is-generating");
          generateBtn.disabled = false;
          cancelBtn.disabled   = true;
        }
      }, delay);
      delay += 120;
    });

    failed.forEach((f) => appendFailedItem(f));
    failedCountEl.textContent = String(failed.length);

    if (!results.length) {
      document.body.classList.remove("is-generating");
      generateBtn.disabled = false;
      cancelBtn.disabled   = true;
      setProgressText(
        failed.length
          ? "All URLs failed to process."
          : "No comments returned."
      );
      setProgressRatio(1);
    }
  } catch (err) {
    console.error("Generate error", err);
    document.body.classList.remove("is-generating");
    generateBtn.disabled = false;
    cancelBtn.disabled   = true;
    setProgressText("Error contacting CrownTALK backend. Please try again.");
    setProgressRatio(0);
  }
}

// ------------------------
// Cancel & Clear
// ------------------------
function handleCancel() {
  cancelled = true;
  document.body.classList.remove("is-generating");
  generateBtn.disabled = false;
  cancelBtn.disabled   = true;
  setProgressText("Cancelled.");
  setProgressRatio(0);
}

function handleClear() {
  urlInput.value = "";
  resetResults();
  resetProgress();
  autoResizeTextarea();
}

// ------------------------
// Reroll
// ------------------------
async function handleReroll(tweetEl) {
  const url = tweetEl?.dataset.url;
  if (!url) return;

  const button = tweetEl.querySelector(".reroll-btn");
  if (!button) return;

  const oldLabel = button.textContent;
  button.disabled = true;
  button.textContent = "Rerolling…";

  const commentsWrap = tweetEl.querySelector(".comments");
  if (commentsWrap) {
    commentsWrap.innerHTML = "";
    const sk1 = document.createElement("div"); sk1.className = "tweet-skeleton-line";
    const sk2 = document.createElement("div"); sk2.className = "tweet-skeleton-line";
    commentsWrap.appendChild(sk1); commentsWrap.appendChild(sk2);
  }

  try {
    const res = await fetch(rerollURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error(`Reroll failed: ${res.status}`);

    const data = await res.json();
    if (data && Array.isArray(data.comments)) {
      updateTweetBlock(tweetEl, { url: data.url || url, comments: data.comments });
    } else {
      setProgressText("Reroll failed for this tweet.");
    }
  } catch (err) {
    console.error("Reroll error", err);
    setProgressText("Network error during reroll.");
  } finally {
    button.disabled = false;
    button.textContent = oldLabel;
  }
}

// ------------------------
// Theme
// ------------------------
const THEME_STORAGE_KEY = "crowntalk_theme";
const ALLOWED_THEMES = ["white","dark-purple","gold","blue","black","emerald","crimson"];

function sanitizeThemeDots() {
  themeDots.forEach((dot) => {
    if (!dot?.dataset) return;
    const t = (dot.dataset.theme || "").trim();
    if (t === "texture") {
      dot.parentElement && dot.parentElement.removeChild(dot);
    } else if (!ALLOWED_THEMES.includes(t)) {
      dot.dataset.theme = "crimson";
    }
  });
  themeDots = Array.from(document.querySelectorAll(".theme-dot"));
}

function applyTheme(themeName) {
  const html = document.documentElement;
  const t = ALLOWED_THEMES.includes(themeName) ? themeName : "dark-purple";
  html.setAttribute("data-theme", t);
  themeDots.forEach((dot) =>
    dot.classList.toggle("is-active", dot.dataset.theme === t)
  );
  try {
    localStorage.setItem(THEME_STORAGE_KEY, t);
  } catch {}
}

function initTheme() {
  sanitizeThemeDots();
  let theme = "dark-purple";
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) {
      theme = stored === "neon" ? "crimson" : stored;
      if (stored === "neon") localStorage.setItem(THEME_STORAGE_KEY, "crimson");
    }
    if (!ALLOWED_THEMES.includes(theme)) theme = "dark-purple";
  } catch {}
  applyTheme(theme);
}

/* ---------- Boot UI once unlocked ---------- */
function bootAppUI() {
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());
  initTheme();
  renderHistory();
  autoResizeTextarea();

  // Gentle warmup shortly after UI becomes usable
  setTimeout(() => {
    maybeWarmBackend();
  }, 4000);

  urlInput?.addEventListener("input", autoResizeTextarea);

  generateBtn?.addEventListener("click", () => {
    if (!document.body.classList.contains("is-generating")) handleGenerate();
  });

  cancelBtn?.addEventListener("click", handleCancel);
  clearBtn?.addEventListener("click", handleClear);

  clearHistoryBtn?.addEventListener("click", () => {
    historyItems = [];
    renderHistory();
  });

  resultsEl?.addEventListener("click", async (event) => {
    const copyBtn  = event.target.closest(".copy-btn, .copy-btn-en");
    const rerollBtn = event.target.closest(".reroll-btn");

    if (copyBtn) {
      const text = copyBtn.dataset.text || "";
      if (!text) return;
      await copyToClipboard(text);
      addToHistory(text);

      const line = copyBtn.closest(".comment-line");
      if (line) line.classList.add("copied");
      copyBtn.classList.add("is-copied");

      const old = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      copyBtn.disabled = true;
      setTimeout(() => {
        copyBtn.textContent = old;
        copyBtn.disabled = false;
      }, 700);
    }

    if (rerollBtn) {
      const tweetEl = rerollBtn.closest(".tweet");
      if (tweetEl) handleReroll(tweetEl);
    }
  });

  historyEl?.addEventListener("click", async (event) => {
    const btn = event.target.closest(".history-copy-btn");
    if (!btn) return;
    const text = btn.dataset.text || "";
    if (!text) return;

    await copyToClipboard(text);
    const old = btn.textContent;
    btn.textContent = "Copied";
    btn.disabled = true;
    setTimeout(() => {
      btn.textContent = old;
      btn.disabled = false;
    }, 700);
  });

  themeDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const t = dot.dataset.theme;
      if (t) applyTheme(t);
    });
  });
}


// Keep the backend warm while the tab is open (no effect if tab is hidden)
(function keepAliveWhileVisible() {
  const PING_MS = 4 * 60 * 1000; // every 4 minutes
  let timer = null;

  function schedule() {
    clearInterval(timer);
    if (document.visibilityState === "visible") {
      timer = setInterval(() => {
        fetch("https://crowntalk.onrender.com/ping", {
          method: "GET",
          cache: "no-store",
          mode: "no-cors",
          keepalive: true,
        }).catch(() => {});
      }, PING_MS);
    }
  }

  document.addEventListener("visibilitychange", schedule);
  window.addEventListener("pagehide", () => clearInterval(timer));
  schedule();
})();

/* =========================================================
   CrownTALK – Premium Lite JS
   Safe to append at end of script.js
========================================================= */

(function(){
  const $ = (sel,root=document)=>root.querySelector(sel);
  const $$ = (sel,root=document)=>[...root.querySelectorAll(sel)];

  // ---------- [10] Ambient progress: flag first card while generating ----------
  const firstCard = $('.card');
  const body = document.body;
  const obs = new MutationObserver(() => {
    if (body.classList.contains('is-generating')) firstCard?.classList.add('is-ambient-progress');
    else firstCard?.classList.remove('is-ambient-progress');
  });
  obs.observe(body,{attributes:true, attributeFilter:['class']});

  // ---------- [3] Dense view toggle (UI inject into Results header) ----------
  (function addDensityToggle(){
    const resultsHeader = $$('.card .card-head')[1] || $('.card-head'); // results card
    if (!resultsHeader || resultsHeader.querySelector('.ct-density-toggle')) return;
    const wrap = document.createElement('div');
    wrap.className = 'ct-density-toggle';
    wrap.innerHTML = `
      <span>Dense</span>
      <button class="ct-switch" type="button" aria-pressed="false"></button>
    `;
    resultsHeader.appendChild(wrap);
    const btn = wrap.querySelector('.ct-switch');
    const apply = (on)=>{ document.body.classList.toggle('ct-dense', !!on); btn.setAttribute('aria-pressed', on?'true':'false'); };
    // restore
    apply(localStorage.getItem('ct_dense') === '1');
    btn.addEventListener('click', ()=>{
      const on = !(localStorage.getItem('ct_dense') === '1');
      localStorage.setItem('ct_dense', on?'1':'0'); apply(on);
    });
  })();

  // ---------- Processing pipeline (static badges) ----------
  (function addPipeline(){
    const row = $('.progress-row');
    if (!row || row.querySelector('.ct-pipe')) return;
    const el = document.createElement('div');
    el.className = 'ct-pipe';
    el.innerHTML = `
      <span class="ct-step"><i class="ct-dot"></i>Fetch</span>
      <span class="ct-step"><i class="ct-dot"></i>Parse</span>
      <span class="ct-step"><i class="ct-dot"></i>Generate</span>
      <span class="ct-step"><i class="ct-dot"></i>Validate</span>
      <span class="ct-step"><i class="ct-dot"></i>Render</span>`;
    row.appendChild(el);
    // Light the steps at rough times while generating (very cheap)
    let timer = null;
    const steps = $$('.ct-step', el);
    const setOn = (n)=>steps.forEach((s,i)=>s.classList.toggle('is-on', i<=n));
    const pump = ()=>{
      setOn(0); timer = setTimeout(()=>{ setOn(1); timer = setTimeout(()=>{ setOn(2); timer = setTimeout(()=>{ setOn(3); timer = setTimeout(()=>setOn(4), 400); }, 600); }, 800); }, 300);
    };
    const mo = new MutationObserver(()=>{
      clearTimeout(timer);
      if (body.classList.contains('is-generating')) pump(); else setOn(-1);
    });
    mo.observe(body,{attributes:true, attributeFilter:['class']});
  })();

  // ---------- Auto low-motion (safe-mode) ----------
  (function autoLowMotion(){
    const prefers = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const weakHW = (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4) ||
                   (navigator.deviceMemory && navigator.deviceMemory <= 4);
    let low = prefers || weakHW;
    // quick FPS probe
    if (!low) {
      let t=performance.now(), frames=0;
      function tick(now){
        frames++; if (now - t < 1000) return requestAnimationFrame(tick);
        const fps = frames; if (fps < 45) low = true;
        if (low) document.body.classList.add('low-motion');
      }
      requestAnimationFrame(tick);
    } else {
      document.body.classList.add('low-motion');
    }
    // tiny pill to toggle off/on
    if (!$('#ctLowPill')) {
      const pill = document.createElement('button');
      pill.id = 'ctLowPill';
      pill.className = 'ct-lowmotion-pill';
      pill.textContent = 'Low-motion';
      pill.title = 'Click to toggle low-motion for this session';
      pill.addEventListener('click', ()=> document.body.classList.toggle('low-motion'));
      document.body.appendChild(pill);
    }
  })();

  // ---------- Session restore (URLs + theme + dense) ----------
  (function sessionStore(){
    const input = $('#urlInput');
    if (!input) return;
    const THEME_KEY = 'ct_theme';
    const URLS_KEY  = 'ct_urls';
    const DENSE_KEY = 'ct_dense';

    // Save URLs as you type (debounced)
    let h=null;
    input.addEventListener('input', ()=>{
      clearTimeout(h); h=setTimeout(()=>localStorage.setItem(URLS_KEY, input.value), 250);
      if (document.body.classList.contains('ct-compact-on')) updateMirror();
    });

    // When theme switches (your theme dots already set data-theme on <html>)
    $$('.theme-dot').forEach(dot=>{
      dot.addEventListener('click', ()=> {
        const th = dot.getAttribute('data-theme');
        localStorage.setItem(THEME_KEY, th);
      });
    });

    // Insert “Restore session” link by Generate row (only if data exists)
    (function addRestore(){
      if (!localStorage.getItem(URLS_KEY)) return;
      const controls = $('.controls'); if (!controls || $('#ctRestore')) return;
      const b = document.createElement('button');
      b.id='ctRestore'; b.className='btn-secondary subtle'; b.textContent='Restore session';
      b.style.marginLeft = 'auto';
      b.addEventListener('click', ()=>{
        const urls = localStorage.getItem(URLS_KEY) || '';
        input.value = urls; input.dispatchEvent(new Event('input'));
        const t = localStorage.getItem(THEME_KEY);
        if (t) document.documentElement.setAttribute('data-theme', t);
        if (localStorage.getItem(DENSE_KEY)==='1') document.body.classList.add('ct-dense'); else document.body.classList.remove('ct-dense');
      });
      controls.appendChild(b);
    })();
  })();

  // ---------- [13] URL “compact view” (mirror overlay) ----------
  (function urlCompact(){
    const container = $('.ai-chat-input .input-section');
    const input = $('#urlInput');
    if (!container || !input) return;

    // Toggle button
    const btn = document.createElement('button');
    btn.id = 'ctUrlCompactToggle';
    btn.type = 'button';
    btn.title = 'Toggle compact URLs (visual only)';
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M3 12h18M3 6h18M3 18h18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
    container.appendChild(btn);

    // Mirror
    const mirror = document.createElement('pre');
    mirror.id = 'ctUrlMirror';
    container.appendChild(mirror);

    const shorten = (line)=>{
      // keep domain + /status/… tail
      try{
        const url = new URL(line.trim());
        const segs = url.pathname.split('/').filter(Boolean);
        if (segs.length >= 2){
          const user = segs[0];
          const tail = segs.slice(-1)[0];
          return `https://${url.host}/${user}/…/${tail}`;
        }
        return `https://${url.host}${url.pathname}`;
      }catch(_){ return line.length > 56 ? line.slice(0, 32) + '…' + line.slice(-18) : line; }
    };
    const updateMirror = ()=>{
      const lines = input.value.split(/\r?\n/).map(l=>l.trim()).filter(Boolean);
      mirror.textContent = lines.map((l,i)=>`${i+1}. ${shorten(l)}`).join('\n');
      mirror.scrollTop = input.scrollTop; // track scroll
    };
    input.addEventListener('input', updateMirror);
    input.addEventListener('scroll', ()=>{ mirror.scrollTop = input.scrollTop; });

    btn.addEventListener('click', ()=>{
      document.body.classList.toggle('ct-compact-on');
      if (document.body.classList.contains('ct-compact-on')) updateMirror();
    });

    // expose for session module
    window.updateMirror = updateMirror;
  })();

  // ---------- [14] Micro confetti after 10 copies (desktop, not low-motion) ----------
  (function copyConfetti(){
    if (matchMedia('(pointer:coarse)').matches) return; // skip touch
    document.addEventListener('click', (e)=>{
      const t = e.target.closest('.copy-btn,.copy-btn-en,.history-copy-btn');
      if (!t) return;
      const k = 'ct_copy_count';
      const n = (parseInt(sessionStorage.getItem(k)||'0',10)+1);
      sessionStorage.setItem(k, n);
      if (n === 10 && !document.body.classList.contains('low-motion')) {
        const wrap = document.createElement('div'); wrap.className='ct-confetti';
        wrap.innerHTML = '<i style="--dx:-60px; --dy:-88px"></i><i style="--dx:66px; --dy:-84px"></i><i style="--dx:-88px; --dy:-62px"></i><i style="--dx:84px; --dy:-72px"></i><i style="--dx:-70px; --dy:-90px"></i><i style="--dx:60px; --dy:-68px"></i>';
        document.body.appendChild(wrap);
        setTimeout(()=>wrap.remove(), 750);
      }
    });
  })();

  // ---------- Theme hover preview on hero ----------
  (function heroPreview(){
    const hero = $('.hero'); if (!hero) return;
    $$('.theme-dot').forEach(dot=>{
      const th = dot.getAttribute('data-theme');
      dot.addEventListener('mouseenter', ()=> hero.setAttribute('data-preview-theme', th));
      dot.addEventListener('mouseleave', ()=> hero.removeAttribute('data-preview-theme'));
    });
  })();

})();

/* ===============================
   CrownTALK Premium Patch Pack
   Features: 16,19,21,22,27,29
   No renames; all hooks are additive
=================================*/
(function () {
  // ---------- small helpers ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const on = (el, ev, cb, opts) => el && el.addEventListener(ev, cb, opts);
  const storage = {
    get(k, d = null) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set(k, v) { localStorage.setItem(k, JSON.stringify(v)); }
  };

  // try to find your existing elements (names from your baseline)
  const textarea = document.querySelector('.url-textarea, .ai-chat-input .main-input');
  const resultsContainer = document.getElementById('results') || document.querySelector('.results-list');
  const progressFill = document.getElementById('progressBarFill');
  const generateBtn = document.getElementById('generateBtn') || document.querySelector('.controls .btn');
  const cancelBtn = document.getElementById('cancelBtn');

  // ---------------------------------------------------
  // 16) Smart Paste+  (sanitize, dedupe, normalize)
  // ---------------------------------------------------
  const URL_RE = /https?:\/\/(?:www\.)?(?:x|twitter)\.com\/[^ \n]+/gi;

  function normalizeX(url) {
    try {
      const u = new URL(url.trim());
      // normalize twitter -> x and strip tracking/query/hash
      if (u.hostname === 'twitter.com') u.hostname = 'x.com';
      u.search = ''; u.hash = '';
      // remove trailing slash variations like .../status/1234/
      u.pathname = u.pathname.replace(/\/+$/, '');
      return u.toString();
    } catch { return null; }
  }

  function smartPasteTransform(text) {
    // grab links, normalize, dedupe, keep numeric ordering if prefixed 1., 2., etc.
    const lines = text.split(/\r?\n/);
    const out = [];
    const seen = new Set();

    for (let raw of lines) {
      const matches = raw.match(URL_RE);
      if (!matches) { out.push(raw); continue; }
      let replaced = raw;
      for (const m of matches) {
        const clean = normalizeX(m);
        if (!clean) continue;
        replaced = replaced.replace(m, clean);
      }
      out.push(replaced);
    }

    // re-scan and compile unique URL list preserving line order
    const flat = out.join('\n').match(URL_RE) || [];
    const uniq = [];
    for (const u of flat) {
      const n = normalizeX(u);
      if (n && !seen.has(n)) { seen.add(n); uniq.push(n); }
    }

    // if content looks like “only URLs”, return enumerated nice list; otherwise return normalized text
    const onlyUrls = lines.every(l => l.trim() === '' || URL_RE.test(l));
    if (onlyUrls) {
      return uniq.map((u, i) => `${i + 1}. ${u}`).join('\n');
    }
    return out.join('\n');
  }

  if (textarea) {
    // on paste -> transform input
    on(textarea, 'paste', (e) => {
      const items = (e.clipboardData || window.clipboardData);
      if (!items) return;
      const text = items.getData('text/plain');
      if (!text) return;
      // prevent default and inject cleaned text
      e.preventDefault();
      const cleaned = smartPasteTransform(text);
      const [start, end] = [textarea.selectionStart, textarea.selectionEnd];
      const before = textarea.value.slice(0, start);
      const after = textarea.value.slice(end);
      textarea.value = before + cleaned + after;
      const pos = (before + cleaned).length;
      textarea.setSelectionRange(pos, pos);
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }

  // ---------------------------------------------------
  // Memory-Aware Re-roll (session + localStorage)
  // ---------------------------------------------------
  const MEMORY_KEY = 'ct_comment_memory_v1';
  const memory = new Set(storage.get(MEMORY_KEY, []));

  function remember(line) {
    if (!line) return;
    memory.add(line.trim());
    storage.set(MEMORY_KEY, Array.from(memory));
  }

  // hook copy buttons in results to remember copied phrases
  function wireCopyMemory(root = document) {
    $$('.copy-btn, .copy-btn-en, .history-copy-btn', root).forEach(btn => {
      if (btn._ct_mem) return;
      btn._ct_mem = true;
      on(btn, 'click', () => {
        const row = btn.closest('.comment-line');
        const text = row?.querySelector('.comment-text')?.textContent?.trim();
        remember(text);
      });
    });
  }
  wireCopyMemory(document);
  // MutationObserver to pick up new result rows
  const mo = new MutationObserver(m => m.forEach(rec => rec.addedNodes.forEach(n => wireCopyMemory(n))));
  if (resultsContainer) mo.observe(resultsContainer, { childList: true, subtree: true });

  // public hook for your existing re-roll: filter out memorized phrases
  // call `window.ctFilterGenerated(commentsArray)` before rendering
  window.ctFilterGenerated = function (arr) {
    if (!Array.isArray(arr)) return arr;
    return arr.filter(txt => !memory.has((txt || '').trim()));
  };

  // ---------------------------------------------------
  // Session Save/Restore (inputs, results, theme)
  // ---------------------------------------------------
  const SESS_KEY = 'ct_sessions_v1';
  const sessions = storage.get(SESS_KEY, {});

  function makeSessionSnapshot() {
    return {
      ts: Date.now(),
      theme: document.documentElement.getAttribute('data-theme') || 'dark-purple',
      input: textarea ? textarea.value : '',
      results: (resultsContainer ? resultsContainer.innerHTML : ''),
      flags: {
        lowMotion: document.body.classList.contains('low-motion'),
        focusMode: document.body.classList.contains('focus-mode')
      }
    };
  }

  function saveSession(name) {
    if (!name) return;
    sessions[name] = makeSessionSnapshot();
    storage.set(SESS_KEY, sessions);
    hudToast(`Saved session “${name}”`);
  }

  function loadSession(name) {
    const s = sessions[name];
    if (!s) return hudToast(`No session “${name}”`, 'warn');
    document.documentElement.setAttribute('data-theme', s.theme);
    if (textarea) textarea.value = s.input || '';
    if (resultsContainer) resultsContainer.innerHTML = s.results || '';
    document.body.classList.toggle('low-motion', !!s.flags?.lowMotion);
    document.body.classList.toggle('focus-mode', !!s.flags?.focusMode);
    hudToast(`Loaded session “${name}”`);
  }

  // expose minimal API (you can bind to UI later)
  window.ctSessions = {
    list: () => Object.keys(sessions).sort(),
    save: saveSession,
    load: loadSession,
    remove: (name) => { delete sessions[name]; storage.set(SESS_KEY, sessions); },
  };

  // quick keyboard helpers:
  // Ctrl/Cmd+S => save prompt
  // Ctrl/Cmd+O => load prompt
  on(document, 'keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    if (e.key.toLowerCase() === 's') {
      e.preventDefault();
      const name = prompt('Save session as:');
      if (name) saveSession(name.trim());
    } else if (e.key.toLowerCase() === 'o') {
      e.preventDefault();
      const keys = Object.keys(sessions);
      if (!keys.length) return hudToast('No saved sessions yet', 'warn');
      const name = prompt(`Load which session?\n${keys.join('\n')}`);
      if (name) loadSession(name.trim());
    }
  });

  // ---------------------------------------------------
  // Rate-Safe Scheduler (batch drip)
  // ---------------------------------------------------
  const scheduler = (function () {
    // simple token bucket-ish throttle
    const cfg = {
      perMinute: 25,      // max requests per minute (tune for your backend/api)
      concurrency: 1,     // one at a time to keep UI simple
      retryBackoffMs: 2500
    };

    let queue = [];
    let running = 0;
    let tickTimestamps = [];

    function canFire() {
      const now = Date.now();
      tickTimestamps = tickTimestamps.filter(t => now - t < 60_000);
      return tickTimestamps.length < cfg.perMinute && running < cfg.concurrency;
    }

    async function runOne(job) {
      running++;
      tickTimestamps.push(Date.now());
      try {
        const t0 = performance.now();
        const res = await job.fn();
        const dt = performance.now() - t0;
        hudUpdate({ lastLatency: dt });
        job.resolve(res);
      } catch (e) {
        if (job.retries > 0) {
          job.retries--;
          setTimeout(() => { queue.push(job); pump(); }, cfg.retryBackoffMs);
        } else {
          job.reject(e);
        }
      } finally {
        running--;
        pump();
      }
    }

    function pump() {
      while (queue.length && canFire()) {
        const job = queue.shift();
        runOne(job);
      }
      hudUpdate();
    }

    function schedule(fn, desc = 'job', retries = 1) {
      return new Promise((resolve, reject) => {
        queue.push({ fn, desc, retries, resolve, reject });
        pump();
      });
    }

    // small HUD hooks
    function info() {
      const now = Date.now();
      tickTimestamps = tickTimestamps.filter(t => now - t < 60_000);
      const margin = Math.max(cfg.perMinute - tickTimestamps.length, 0);
      return { queue: queue.length + running, margin, rate: cfg.perMinute, running };
    }

    return { schedule, info, cfg };
  })();

  window.ctSchedule = scheduler; // expose for your generate loop if you want

  // Example integration:
  // wherever you fire fetch to /comment or /reroll, wrap with scheduler.schedule(() => fetch(...))
  // That’s all—no renames required.

  // ---------------------------------------------------
  // Live System HUD
  // ---------------------------------------------------
  const hud = (function () {
    const el = document.createElement('div');
    el.id = 'ctHud';
    el.innerHTML = `
      <div class="ct-hud__row">
        <span class="ct-hud__dot"></span>
        <strong>HUD</strong>
        <span class="ct-hud__sp">·</span>
        <span id="ctHudQueue">Q:0</span>
        <span id="ctHudRate">/min:–</span>
        <span id="ctHudMargin">free:–</span>
        <span id="ctHudLatency">lat:–</span>
      </div>
      <div id="ctHudToast" class="ct-hud__toast" aria-live="polite"></div>
    `;
    document.body.appendChild(el);
    const state = { lastLatency: null };
    function update(extra = {}) {
      Object.assign(state, extra);
      const inf = scheduler.info();
      $('#ctHudQueue').textContent = `Q:${inf.queue}`;
      $('#ctHudRate').textContent = `/min:${inf.rate}`;
      $('#ctHudMargin').textContent = `free:${inf.margin}`;
      $('#ctHudLatency').textContent = state.lastLatency ? `lat:${Math.round(state.lastLatency)}ms` : 'lat:–';
      el.classList.toggle('is-hot', inf.margin < Math.max(2, Math.round(inf.rate * 0.1)));
    }
    function toast(msg, kind = 'ok') {
      const t = $('#ctHudToast');
      t.textContent = msg;
      t.dataset.kind = kind;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 1800);
    }
    return { update, toast, el };
  })();

  function hudUpdate(extra) { hud.update(extra); }
  function hudToast(msg, kind) { hud.toast(msg, kind); }
  hudUpdate();

  // ---------------------------------------------------
  // Accessibility & Focus Mode
  //   - Focus mode dims everything except active card
  //   - Low-motion toggle (reduces animations)
  // ---------------------------------------------------
  // keyboard:
  //   F => toggle focus mode
  //   M => toggle low-motion
  on(document, 'keydown', (e) => {
    if (e.target && /input|textarea/i.test(e.target.tagName)) {
      if (e.key.toLowerCase() === 'f' && (e.ctrlKey || e.metaKey)) { // Ctrl/Cmd+F would clash; use plain 'f'
        return;
      }
    }
    if (e.key.toLowerCase() === 'f') {
      document.body.classList.toggle('focus-mode');
      hudToast(`Focus mode: ${document.body.classList.contains('focus-mode') ? 'on' : 'off'}`);
    }
    if (e.key.toLowerCase() === 'm') {
      document.body.classList.toggle('low-motion');
      hudToast(`Low-motion: ${document.body.classList.contains('low-motion') ? 'on' : 'off'}`);
    }
  });

  // when textarea/card gains focus, mark that card as active (for focus mode visuals)
  $$('.card').forEach(card => {
    on(card, 'focusin', () => card.classList.add('is-focused'));
    on(card, 'focusout', () => card.classList.remove('is-focused'));
  });

  // ---------------------------------------------------
  // Remove the preview button near the input (if present)
  // (Earlier “compact preview” toggle was #ctUrlCompactToggle. Hide & detach events.)
  // ---------------------------------------------------
  (function killPreviewButton() {
    const btn = document.getElementById('ctUrlCompactToggle') || document.querySelector('.ct-preview-toggle, .url-preview-btn');
    if (btn) {
      btn.remove(); // hard remove so it won’t occupy space
    }
  })();

})();

// Remove Dense toggle button and neutralize any leftover dense state
document.querySelectorAll('.ct-density-toggle, .ct-switch').forEach(el => el.remove());
document.body.classList.remove('ct-dense');

/* CrownTALK — Progress helper (indeterminate + determinate)
   Works with:
     - body.is-generating
     - #progressBarFill (your existing inner bar)
   Public API: ctProgress.start(), ctProgress.step(pct),
               ctProgress.done(), ctProgress.cancel()
*/
(function () {
  const BODY = document.body;
  const FILL = document.getElementById('progressBarFill');
  let finishTimer = null;

  function clamp01(x){ return Math.max(0, Math.min(1, x)); }
  function setGenerating(on){
    BODY.classList.toggle('is-generating', !!on);
    if (on) {
      // if caller forgets to set width, ensure there's *some* visible fill
      if (FILL && (!FILL.style.width || FILL.style.width === '0%')) FILL.style.width = '8%';
    }
  }
  function flashDone(ms = 420){
    BODY.classList.add('ct-progress-done');
    clearTimeout(finishTimer);
    finishTimer = setTimeout(() => BODY.classList.remove('ct-progress-done'), ms);
  }

  window.ctProgress = {
    // Start the animated state; optionally seed a starting width (0–100)
    start(initialPct = 0){
      setGenerating(true);
      if (FILL && typeof initialPct === 'number') {
        FILL.style.width = (clamp01(initialPct / 100) * 100).toFixed(2) + '%';
      }
    },

    // Update bar width during work (0–100). Safe to call often.
    step(pct){
      if (!FILL || typeof pct !== 'number') return;
      FILL.style.width = (clamp01(pct / 100) * 100).toFixed(2) + '%';
    },

    // Stop the animated state. Options:
    //   flash: play green/orange finish sweep (default true)
    //   resetWidth: reset inner bar back to 0% (default true)
    done({ flash = true, resetWidth = true } = {}){
      setGenerating(false);
      if (flash) flashDone();
      if (resetWidth && FILL) FILL.style.width = '0%';
    },

    // Cancel immediately (no finish flash)
    cancel(){
      setGenerating(false);
      if (FILL) FILL.style.width = '0%';
    }
  };
})();


/* CrownTALK — Desktop Premium Patch v1 (JS)
   Desktop-only. No HTML changes required. */
(function () {
  if (!matchMedia('(pointer:fine) and (hover:hover)').matches) return;

  const $  = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => [...r.querySelectorAll(s)];

  /* ---------- Magnetic hover (tiny) ---------- */
  function addMagnetic(el, amt = 6) {
    if (!el || el._ctMag) return; el._ctMag = true;
    let raf = 0;
    el.addEventListener('mousemove', e => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width  - .5) * amt;
      const y = ((e.clientY - r.top)  / r.height - .5) * amt;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        el.style.setProperty('--ct-tx', x.toFixed(2) + 'px');
        el.style.setProperty('--ct-ty', y.toFixed(2) + 'px');
      });
    });
    el.addEventListener('mouseleave', () => {
      el.style.setProperty('--ct-tx', '0px');
      el.style.setProperty('--ct-ty', '0px');
    });
  }
  ['#generateBtn', '.reroll-btn', '.card'].forEach(sel => $$(sel).forEach(addMagnetic));

  /* ---------- Cursor snap tooltip ---------- */
  const tip = document.createElement('div');
  tip.id = 'ctTip';
  document.body.appendChild(tip);

  function attachTip(el, text) {
    if (!el || el.dataset.ctTip) return;
    el.dataset.ctTip = text;
    el.addEventListener('mousemove', (e) => {
      tip.textContent = el.dataset.ctTip || '';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top  = (e.clientY + 12) + 'px';
      tip.classList.add('show');
    });
    const hide = () => tip.classList.remove('show');
    el.addEventListener('mouseleave', hide);
    el.addEventListener('blur', hide);
  }
  attachTip($('#generateBtn'), 'Generate');
  $$('.reroll-btn').forEach(b => attachTip(b, 'Reroll'));
  $$('.copy-btn,.copy-btn-en').forEach(b => attachTip(b, 'Copy'));

  // results are dynamic: observe and attach tips + magnetic
  const results = $('#results');
  if (results) {
    new MutationObserver(muts => muts.forEach(mu => mu.addedNodes.forEach(n => {
      if (!(n instanceof HTMLElement)) return;
      n.querySelectorAll?.('.reroll-btn').forEach(b => { attachTip(b, 'Reroll'); addMagnetic(b, 5); });
      n.querySelectorAll?.('.copy-btn,.copy-btn-en').forEach(b => attachTip(b, 'Copy'));
      if (n.matches?.('.card')) addMagnetic(n, 4);
    }))).observe(results, {childList: true, subtree: true});
  }

  /* ---------- Focus halo: rely on your existing .is-focused hooks ---------- */
  $$('.card').forEach(card => {
    card.addEventListener('focusin',  () => card.classList.add('is-focused'));
    card.addEventListener('focusout', () => card.classList.remove('is-focused'));
  });

  /* ---------- Premium modals (welcome & empty-generate) ---------- */
  let modal;
  function ensureModal() {
    if (modal) return modal;
    modal = document.createElement('div');
    modal.className = 'ct-modal';
    modal.innerHTML = `
      <div class="ct-card" role="dialog" aria-modal="true" aria-labelledby="ctTitle">
        <h3 id="ctTitle"></h3>
        <p id="ctBody"></p>
        <div class="ct-actions">
          <button type="button" class="btn-ghost" id="ctCancel">Close</button>
          <button type="button" class="btn-primary" id="ctOk">OK</button>
        </div>
      </div>`;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) hideModal(); });
    modal.querySelector('#ctCancel').addEventListener('click', hideModal);
    modal.querySelector('#ctOk').addEventListener('click', () => {
      if (modal.dataset.kind === 'empty') $('#urlInput')?.focus();
      hideModal();
    });
    return modal;
  }
  function showModal(kind) {
    ensureModal();
    modal.dataset.kind = kind;
    const title = modal.querySelector('#ctTitle');
    const body  = modal.querySelector('#ctBody');
    const ok    = modal.querySelector('#ctOk');

    if (kind === 'welcome') {
      title.textContent = 'Welcome to CrownTALK ✨';
      body.textContent  = 'Paste one or more tweet links, press Generate, then copy your favorites. Smart Paste+ cleans and dedupes automatically.';
      ok.textContent    = 'Let’s go';
    } else {
      title.textContent = 'Nothing to generate';
      body.textContent  = 'Add at least one tweet URL to get started.';
      ok.textContent    = 'Got it';
    }
    modal.classList.add('show');
  }
  function hideModal() { modal?.classList.remove('show'); }

  // daily welcome on desktop (first visit each day)
  (function dailyWelcome(){
    const k = (()=>{ const d=new Date(); return `ct_welcome_${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`; })();
    if (!localStorage.getItem(k)) {
      showModal('welcome');
      localStorage.setItem(k, '1');
    }
  })();

  // intercept Generate click when no URLs present (capture phase so it runs before existing handler)
  const genBtn = $('#generateBtn');
  if (genBtn) {
    genBtn.addEventListener('click', (e) => {
      const val = ($('#urlInput')?.value || '').trim();
      if (!val) {
        e.preventDefault(); e.stopImmediatePropagation();
        showModal('empty');
      }
    }, true);
  }

  /* ---------- Optional: mark active theme dot for comet tick ---------- */
  // If your applyTheme already toggles .is-active, nothing to do. If not, keep this tiny guard:
  (function ensureActiveDot(){
    const htmlTheme = document.documentElement.getAttribute('data-theme');
    const active = $(`.theme-dot[data-theme="${htmlTheme}"]`);
    if (active && !active.classList.contains('is-active')) {
      $$('.theme-dot').forEach(d => d.classList.remove('is-active'));
      active.classList.add('is-active');
    }
  })();
})();


/* CrownTALK — Desktop Premium Animation Pack */
(function(){
  const isDesktop = matchMedia('(pointer:fine) and (hover:hover)').matches;
  const lowMotion = () => document.body.classList.contains('low-motion') ||
                          matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!isDesktop) return;

  const $  = (s, r=document)=>r.querySelector(s);

  /* -------- Live favicon spinner while generating (THROTTLED) -------- */
(function faviconSpinnerThrottled(){
  const isDesktop = matchMedia('(pointer:fine) and (hover:hover)').matches;
  if (!isDesktop) return;

  const lowMotion = () =>
    document.body.classList.contains('low-motion') ||
    matchMedia('(prefers-reduced-motion: reduce)').matches;

  let link = document.querySelector('link[rel~="icon"]');
  if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link); }

  const originalHref = link.href || '';
  const cvs = document.createElement('canvas');
  const ctx = cvs.getContext('2d', { willReadFrequently: false });

  let running = false;
  let raf = 0;
  let angle = 0;
  let lastUpdate = 0;
  let currentBlobURL = null;

  function revokeBlobURL() {
    if (currentBlobURL) { URL.revokeObjectURL(currentBlobURL); currentBlobURL = null; }
  }

  function drawAndSetFavicon(now) {
    // Throttle to ~2 fps (every ~500ms)
    if ((now - lastUpdate) < 500) return;
    lastUpdate = now;

    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const sz = 32 * dpr; cvs.width = cvs.height = sz;

    // draw ring
    ctx.clearRect(0, 0, sz, sz);
    const cx = sz / 2, cy = sz / 2, r = sz * 0.36, lw = Math.max(2 * dpr, 3);
    ctx.globalAlpha = 0.22; ctx.lineWidth = lw; ctx.strokeStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

    // arc
    ctx.globalAlpha = 0.95; ctx.strokeStyle = 'rgba(120,180,255,1)';
    ctx.beginPath(); ctx.arc(cx, cy, r, angle, angle + Math.PI * 1.1); ctx.stroke();

    // glow dot
    const sx = cx + r * Math.cos(angle + Math.PI * 1.1);
    const sy = cy + r * Math.sin(angle + Math.PI * 1.1);
    const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, 4 * dpr);
    grd.addColorStop(0, '#fff'); grd.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grd; ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(sx, sy, 4 * dpr, 0, Math.PI * 2); ctx.fill();

    // set favicon via Blob (lighter than dataURL), cleanup previous
    cvs.toBlob((blob) => {
      if (!blob) return;
      revokeBlobURL();
      currentBlobURL = URL.createObjectURL(blob);
      try { link.href = currentBlobURL; } catch {}
    }, 'image/png');

    angle += 0.35; // slower spin
  }

  function loop(now) {
    if (!running) return;
    // Pause work if tab hidden or low-motion
    if (document.visibilityState === 'visible' && !lowMotion()) {
      drawAndSetFavicon(now || performance.now());
    }
    raf = requestAnimationFrame(loop);
  }

  // watch is-generating flip
  const mo = new MutationObserver(() => {
    const gen = document.body.classList.contains('is-generating');
    if (gen && !running && !lowMotion()) {
      running = true; lastUpdate = 0; loop();
    } else if (!gen || lowMotion()) {
      running = false; cancelAnimationFrame(raf);
      revokeBlobURL();
      if (originalHref) link.href = originalHref;
    }
  });
  mo.observe(document.body, { attributes: true, attributeFilter: ['class'] });

  // also pause when tab hidden; resume when visible
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') {
      running = false; cancelAnimationFrame(raf);
    } else if (document.body.classList.contains('is-generating') && !lowMotion()) {
      running = true; lastUpdate = 0; loop();
    }
  });
})();


  /* -------- Results scroll-edge glow -------- */
  (function scrollHints(){
    const el = $('#results'); if (!el) return;
    const update = ()=>{
      const canUp   = el.scrollTop > 2;
      const canDown = el.scrollHeight - el.clientHeight - el.scrollTop > 2;
      el.classList.toggle('ct-can-scroll-up', canUp);
      el.classList.toggle('ct-can-scroll-down', canDown);
    };
    el.addEventListener('scroll', update, {passive:true});
    if ('ResizeObserver' in window) new ResizeObserver(update).observe(el);
    requestAnimationFrame(update);
  })();

  /* -------- Generate button ripple -------- */
  (function ripple(){
    const btn = $('#generateBtn'); if (!btn) return;
    btn.addEventListener('click', (e)=>{
      if (lowMotion()) return;
      const r = btn.getBoundingClientRect();
      const rip = document.createElement('span');
      rip.className = 'ct-rip';
      rip.style.setProperty('--x', (e.clientX - r.left)+'px');
      rip.style.setProperty('--y', (e.clientY - r.top )+'px');
      btn.appendChild(rip);
      setTimeout(()=>rip.remove(), 620);
    }, {capture:true});
  })();

  /* -------- Copy fly-to-history -------- */
  (function flyToHistory(){
    const history = $('#history'); if (!history) return;
    document.addEventListener('click', (e)=>{
      const btn = e.target.closest?.('.copy-btn,.copy-btn-en,.history-copy-btn'); if (!btn) return;
      if (lowMotion()) return;
      const line = btn.closest('.comment-line'); const txt = (line?.querySelector('.comment-text')?.textContent || 'Copied').trim();
      const pill = document.createElement('div'); pill.className='ct-fly'; pill.textContent = txt.length>28 ? (txt.slice(0,26)+'…') : txt;
      document.body.appendChild(pill);
      const s = btn.getBoundingClientRect();
      const h = history.getBoundingClientRect();
      const startX = s.left + s.width/2, startY = s.top + s.height/2;
      const endX   = h.left + 28,       endY   = h.top + 16;
      pill.style.left = startX+'px'; pill.style.top  = startY+'px';
      requestAnimationFrame(()=>{
        pill.style.transform = `translate3d(${endX-startX}px, ${endY-startY}px, 0)`;
        pill.classList.add('hide');
      });
      setTimeout(()=>pill.remove(), 520);
    });
  })();

  /* -------- Results counter odometer -------- */
  (function odo(){
    const el = $('#resultCount'); if (!el) return;
    let prev = 0;
    const getNum = (t)=> parseInt(String(t).match(/\d+/)?.[0]||'0',10);
    const fmt = (n)=> `${n} tweet${n===1?'':'s'}`;
    const mo = new MutationObserver(()=>{
      const next = getNum(el.textContent);
      if (Number.isNaN(next) || next===prev) { prev = next; return; }
      const start = prev, end = next, dur = 260; let t0=null;
      const span = document.createElement('span'); span.className='ct-odo up'; el.innerHTML=''; el.appendChild(span);
      function tick(ts){ if (!t0) t0 = ts; const p = Math.min((ts-t0)/dur,1); const v = Math.round(start + (end-start)*p); span.textContent = fmt(v); if (p<1) requestAnimationFrame(tick); }
      requestAnimationFrame(tick);
      prev = next;
    });
    mo.observe(el, {childList:true, characterData:true, subtree:true});
  })();
})();

/* ===========================
   CrownTALK — Premium Desk Pack
   (excludes 10,13,14,17)
   Drop at end of script.js
   =========================== */
(function () {
  const desk = matchMedia('(pointer:fine)').matches;
  if (!desk) return;

  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const on = (el, ev, fn, opt) => el && el.addEventListener(ev, fn, opt);

  const results = $('#results');
  const history = $('#history');
  const urlInput = $('#urlInput');
  const generateBtn = $('#generateBtn');
  const progressFill = $('#progressBarFill');
  const progressTrack = progressFill ? progressFill.parentElement : null;

  const lowMotion = () => document.body.classList.contains('low-motion');

  /* --- [20] Time-of-day ambient tint --- */
  (function ambient() {
    const h = (new Date()).getHours();
    const slot = h < 12 ? 'morning' : h < 18 ? 'afternoon' : 'evening';
    document.documentElement.setAttribute('data-ambient', slot);
  })();

  /* --- helpers --- */
  function once(fn) { let ran = false; return (...a) => { if (ran) return; ran = true; fn(...a); }; }

  /* --- [12] Tick marks overlay + progress hook --- */
  let ticks;
  function buildTicks() {
    if (!progressTrack || $('#ctTicks')) return;
    const wrap = document.createElement('div');
    wrap.className = 'ct-ticks';
    wrap.id = 'ctTicks';
    progressTrack.style.position = progressTrack.style.position || 'relative';
    progressTrack.appendChild(wrap);
    // 10%..90%
    for (let i=1;i<=9;i++){
      const t = document.createElement('i');
      t.className = 'tick';
      t.dataset.pos = String(i*10);
      t.style.left = i*10 + '%';
      wrap.appendChild(t);
    }
    ticks = $$('.tick', wrap);
  }
  buildTicks();

  function updateTicks(pct) {
    if (!ticks) return;
    ticks.forEach(t => {
      const pos = parseInt(t.dataset.pos,10);
      t.classList.toggle('passed', pct >= pos);
    });
  }

  // Wrap existing setProgressRatio if present
  (function hookProgress() {
    const old = window.setProgressRatio;
    window.setProgressRatio = function(ratio) {
      if (typeof old === 'function') old(ratio);
      const pct = Math.max(0, Math.min(100, Math.round((Number(ratio)||0)*100)));
      updateTicks(pct);
    };
  })();

  // Support ctProgress.step if you use it
  if (window.ctProgress && typeof window.ctProgress.step === 'function') {
    const origStep = window.ctProgress.step.bind(window.ctProgress);
    window.ctProgress.step = (pct) => { origStep(pct); updateTicks(pct|0); };
  }

  /* --- [3] Scroll-edge ion glow + kinetic scrollbar state --- */
  if (results) {
    const refreshGlow = () => {
      const canUp = results.scrollTop > 0;
      const canDown = results.scrollTop < (results.scrollHeight - results.clientHeight - 1);
      results.classList.toggle('can-up', canUp);
      results.classList.toggle('can-down', canDown);
    };
    const flagScroll = () => {
      results.classList.add('is-scrolling');
      clearTimeout(flagScroll._t);
      flagScroll._t = setTimeout(() => results.classList.remove('is-scrolling'), 180);
    };
    on(results, 'scroll', () => { refreshGlow(); flagScroll(); }, { passive: true });
    setTimeout(refreshGlow, 0);
  }

  /* --- [2] Per-card microbar + [19] ready ping --- */
  function attachCardEffects(card) {
    if (!card || card._ctDecorated) return;
    card._ctDecorated = true;

    // microbar
    const bar = document.createElement('div');
    bar.className = 'ct-microbar';
    card.prepend(bar);
    requestAnimationFrame(() => { bar.classList.add('run'); });

    // observe comments to ping when content appears/changes
    const comments = card.querySelector('.comments');
    if (comments) {
      const m = new MutationObserver(() => {
        if (lowMotion()) return;
        card.classList.add('ct-ready-ping');
        setTimeout(() => card.classList.add('off'), 120);
        setTimeout(() => { card.classList.remove('ct-ready-ping','off'); }, 220);
      });
      m.observe(comments, { childList: true, subtree: true });
    }
  }

  // Observe incoming tweets
  if (results) {
    const obs = new MutationObserver(recs => {
      recs.forEach(r => r.addedNodes.forEach(n => {
        if (n.nodeType === 1 && n.classList.contains('tweet')) attachCardEffects(n);
      }));
    });
    obs.observe(results, { childList:true, subtree:true });
  }

  /* --- [8] Cursor-driven specular on cards --- */
  if (results) {
    let raf = null, target = null, mx = 0, my = 0;
    on(results, 'mousemove', (e) => {
      const t = e.target.closest('.tweet');
      if (!t) return;
      target = t;
      const rect = t.getBoundingClientRect();
      mx = ((e.clientX - rect.left) / rect.width) * 100;
      my = ((e.clientY - rect.top) / rect.height) * 100;
      if (!raf) raf = requestAnimationFrame(() => {
        raf = null;
        target?.style.setProperty('--mx', mx + '%');
        target?.style.setProperty('--my', my + '%');
      });
    });
  }

  /* --- [9] Odometer flick on #resultCount --- */
  (function odometer() {
    const el = $('#resultCount');
    if (!el) return;

    let last = el.textContent;
    const wrapDigits = (txt) => {
      const parts = txt.split(/(\d)/);
      el.innerHTML = '';
      el.classList.add('ct-odometer');
      parts.forEach(p => {
        if (/\d/.test(p)) {
          const d = document.createElement('span');
          d.className = 'ct-digit';
          d.innerHTML = `<span class="front">${p}</span><span class="back">${p}</span>`;
          el.appendChild(d);
        } else {
          const s = document.createElement('span');
          s.textContent = p;
          el.appendChild(s);
        }
      });
    };

    wrapDigits(last);

    const mo = new MutationObserver(() => {
      const now = el.textContent;
      if (now === last) return;
      const olds = Array.from(el.querySelectorAll('.ct-digit .back')).map(n => n.textContent);
      wrapDigits(now);
      const news = el.querySelectorAll('.ct-digit');
      news.forEach((d, i) => {
        const back = d.querySelector('.back');
        const front = d.querySelector('.front');
        if (!back || !front) return;
        // flip only where number changed
        const prev = olds[i] ?? '';
        if (front.textContent !== prev) d.classList.add('flip');
        setTimeout(() => d.classList.remove('flip'), 260);
      });
      last = now;
    });
    mo.observe(el, { characterData:true, subtree:true, childList:true });
  })();

  /* --- [7] Copy heat-streak from clicked copy btn to History --- */
  (function heatStreak() {
    const root = document;
    on(root, 'click', (e) => {
      const btn = e.target.closest('.copy-btn, .copy-btn-en, .history-copy-btn');
      if (!btn || !history) return;
      const a = btn.getBoundingClientRect();
      const b = history.getBoundingClientRect();
      const startX = a.left + a.width/2, startY = a.top + a.height/2;
      const endX = b.left + 12, endY = b.top + 12;

      const dot = document.createElement('div');
      dot.className = 'ct-streak';
      dot.style.transform = `translate(${startX}px, ${startY}px) scale(1)`;
      document.body.appendChild(dot);
      // next frame => fly
      requestAnimationFrame(() => {
        dot.style.transform = `translate(${endX}px, ${endY}px) scale(.6)`;
        setTimeout(() => { dot.style.opacity = 0; setTimeout(()=>dot.remove(), 180); }, 560);
      });
    });
  })();

  /* --- [5] Empty-state premium card (replaces alert) --- */
  (function emptyStateCard() {
    if (!generateBtn || !urlInput) return;
    const show = once(() => {
      const overlay = document.createElement('div');
      overlay.className = 'ct-overlay';
      overlay.id = 'ctEmptyOverlay';
      overlay.innerHTML = `
        <div class="ct-card" role="dialog" aria-modal="true">
          <h3>Paste some X/Twitter links</h3>
          <p>Tip: paste multiple lines — we’ll clean & normalize them automatically.</p>
          <div class="ct-actions">
            <button class="ct-btn" id="ctEmptyDismiss">Cancel</button>
            <button class="ct-btn primary" id="ctEmptyFocus">Got it</button>
          </div>
        </div>`;
      document.body.appendChild(overlay);
      const close = () => overlay.remove();
      on(overlay, 'click', (e)=> { if (e.target === overlay) close(); });
      on($('#ctEmptyDismiss', overlay), 'click', close);
      on($('#ctEmptyFocus', overlay), 'click', ()=>{ close(); urlInput.focus(); });
      on(document, 'keydown', (e)=>{ if (e.key === 'Escape') close(); }, { once:true });
    });

    // override handleGenerate so we intercept cleanly
    const original = window.handleGenerate;
    window.handleGenerate = async function () {
      const val = (urlInput.value || '').trim();
      if (!val) { show(); return; }
      return original?.apply(this, arguments);
    };
  })();

  /* --- [6] Daily welcome card (once per day) --- */
  (function dailyWelcome() {
    if (lowMotion()) return;
    const key = 'ct_welcome_day_v1';
    const today = new Date().toISOString().slice(0,10);
    try {
      const last = localStorage.getItem(key);
      if (last === today) return;
      localStorage.setItem(key, today);
    } catch { /* ignore */ }

    const box = document.createElement('div');
    box.className = 'ct-welcome';
    const greet = (() => {
      const h = (new Date()).getHours();
      return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
    })();
    box.innerHTML = `<div class="title">${greet} 👋</div>
      <div class="hint">Ready to batch? Pro tip: paste multiple links — we’ll handle the rest.</div>`;
    document.body.appendChild(box);

    const kill = () => box.remove();
    setTimeout(kill, 2800);
    on(box, 'click', kill, { once:true });
  })();

  /* --- [18] Laser underline already handled by CSS --- */

  /* --- micro: ensure existing tweets get microbar on load --- */
  $$('.tweet').forEach(attachCardEffects);

  /* --- [1] ensure ticks overlay exists if DOM changes later --- */
  const moProgress = new MutationObserver(buildTicks);
  if (progressTrack) moProgress.observe(progressTrack, { childList:true });

})();


/* ===========================================
   CrownTALK Premium Patch JS
   Features: 4,5,7,8,12,13,19,20
   Desktop-only, motion-aware. No HTML edits.
=========================================== */
(function () {
  const isDesktop = matchMedia('(pointer:fine)').matches;
  const lowMotion = () => document.body.classList.contains('low-motion');

  const $  = (s,r=document)=>r.querySelector(s);
  const $$ = (s,r=document)=>Array.from(r.querySelectorAll(s));

  const urlInput = document.getElementById('urlInput');
  const results  = document.getElementById('results');

  /* ---------- [20] Mini Toast Queue ---------- */
  const toasts = document.createElement('div'); toasts.id = 'ctToasts'; document.body.appendChild(toasts);
  function ctToast(msg, kind='ok', ms=1800) {
    const el = document.createElement('div');
    el.className = 'ct-toast';
    el.dataset.kind = kind;
    el.textContent = msg;
    toasts.appendChild(el);
    const t = setTimeout(() => { el.classList.add('ct-leave'); setTimeout(()=>el.remove(), 260); }, ms);
    el.addEventListener('click', () => { clearTimeout(t); el.classList.add('ct-leave'); setTimeout(()=>el.remove(), 160); });
  }
  window.ctToast = ctToast;

  /* ---------- [8] Ambient Aurora (Generating Only) ---------- */
  const aur = document.createElement('div'); aur.id = 'ctAurora'; document.body.appendChild(aur);

  /* ---------- [13] Tab Title Signals ---------- */
  const originalTitle = document.title;
  let lastPercent = 0;
  function setTitleProcessing(label, pct) {
    document.title = `⏳ CrownTALK — ${label}${Number.isFinite(pct) ? ` (${Math.round(pct)}%)` : ''}`;
  }
  function setTitleDone() {
    document.title = '✅ CrownTALK — done';
    setTimeout(() => { document.title = originalTitle; }, 1200);
  }
  // wrap existing progress helpers if present
  const _setProgressText  = typeof window.setProgressText === 'function' ? window.setProgressText : null;
  const _setProgressRatio = typeof window.setProgressRatio === 'function' ? window.setProgressRatio : null;

  if (_setProgressText) {
    window.setProgressText = function patchedProgressText(txt) {
      try {
        // pick up "Processed X/Y" if available for nicer label
        const m = /Processed\s+(\d+)(?:\/(\d+))?/i.exec(txt || '');
        if (m) {
          const x = parseInt(m[1], 10); const y = parseInt(m[2] || '0', 10);
          setTitleProcessing(`${x}${y?'/'+y:''}`, lastPercent);
        } else {
          setTitleProcessing((txt||'processing'), lastPercent);
        }
      } catch {}
      return _setProgressText.apply(this, arguments);
    };
  }
  if (_setProgressRatio) {
    window.setProgressRatio = function patchedProgressRatio(ratio) {
      lastPercent = Math.max(0, Math.min(100, (ratio||0)*100));
      if (document.body.classList.contains('is-generating')) {
        setTitleProcessing('processing', lastPercent);
      }
      return _setProgressRatio.apply(this, arguments);
    };
  }

  // also observe generating state to flip the title
  const genObs = new MutationObserver(() => {
    if (document.body.classList.contains('is-generating')) {
      setTitleProcessing('processing', lastPercent || 1);
    } else {
      setTitleDone();
    }
  });
  genObs.observe(document.body, { attributes: true, attributeFilter: ['class'] });

  /* ---------- [12] Cursor Trail (Generating Only) ---------- */
  let trail, lastMove = 0;
  function ensureTrail() {
    if (trail) return trail;
    trail = document.createElement('div');
    trail.className = 'ct-trail';
    document.body.appendChild(trail);
    return trail;
  }
  if (isDesktop) {
    window.addEventListener('mousemove', (e) => {
      if (!document.body.classList.contains('is-generating') || lowMotion()) return;
      const now = performance.now();
      if (now - lastMove < 24) return; // throttle ~40fps
      lastMove = now;
      const wrap = ensureTrail();
      const dot1 = document.createElement('div');
      const dot2 = document.createElement('div');
      dot1.className = 'ct-trail-dot';
      dot2.className = 'ct-trail-dot';
      dot1.style.left = dot2.style.left = (e.clientX) + 'px';
      dot1.style.top  = dot2.style.top  = (e.clientY) + 'px';
      wrap.appendChild(dot1); wrap.appendChild(dot2);
      setTimeout(()=>dot1.remove(), 380);
      setTimeout(()=>dot2.remove(), 380);
    }, { passive: true });
  }

  /* ---------- [4] Caret Spark (press Enter) ---------- */
  function caretSpark() {
    if (!urlInput || !isDesktop || lowMotion()) return;
    // compute caret pixel position using a mirror
    const rect = urlInput.getBoundingClientRect();
    const mirror = document.createElement('div');
    mirror.style.cssText = `
      position: fixed; left:${rect.left}px; top:${rect.top}px;
      width:${rect.width}px; padding:${getComputedStyle(urlInput).padding};
      font:${getComputedStyle(urlInput).font};
      line-height:${getComputedStyle(urlInput).lineHeight};
      white-space: pre-wrap; word-wrap: break-word; visibility:hidden;
    `;
    const pre = document.createElement('span');
    const caret = document.createElement('span');
    caret.textContent = '·';
    caret.style.opacity = '0';
    const val = urlInput.value.slice(0, urlInput.selectionStart);
    pre.textContent = val;
    mirror.appendChild(pre); mirror.appendChild(caret);
    document.body.appendChild(mirror);
    const cx = caret.getBoundingClientRect().left;
    const cy = caret.getBoundingClientRect().top;
    mirror.remove();

    const spark = document.createElement('i');
    spark.className = 'ct-caret-spark';
    spark.style.left = (Number.isFinite(cx)? cx : rect.left + 12) + 'px';
    spark.style.top  = (Number.isFinite(cy)? cy : rect.top + 10) + 'px';
    document.body.appendChild(spark);
    setTimeout(()=>spark.remove(), 220);
  }
  if (urlInput) {
    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && isDesktop) caretSpark();
    });
  }

  /* ---------- [5] URL Gutter Lights & [19] Hover Hint Line ---------- */
  (function setupGutter() {
    if (!urlInput || !isDesktop) return;
    const wrapper = urlInput.closest('.ai-chat-input .input-section') || urlInput.parentElement || document.body;
    wrapper.style.position = wrapper.style.position || 'relative';

    const gutter = document.createElement('div'); gutter.className = 'ct-url-gutter';
    const hover  = document.createElement('div'); hover.className  = 'ct-url-hoverline';
    wrapper.appendChild(gutter); wrapper.appendChild(hover);

    const URL_RE = /https?:\/\/(?:www\.)?(?:x|twitter)\.com\/[^ \n]+/i;

    let raf = 0;
    function refresh() {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        gutter.innerHTML = '';
        const lh = parseFloat(getComputedStyle(urlInput).lineHeight) || 22;
        const padTop = parseFloat(getComputedStyle(urlInput).paddingTop) || 12;
        const lines = urlInput.value.split(/\r?\n/);
        // sync gutter height to content height
        for (let i=0;i<lines.length;i++){
          const led = document.createElement('i'); led.className = 'ct-led';
          const t = lines[i].trim();
          if (!t) { led.classList.add('warn'); }
          else if (URL_RE.test(t)) { led.classList.add('ok'); }
          else { led.classList.add('err'); }
          led.style.position = 'absolute';
          led.style.top = (padTop + i*lh - 2 + (-urlInput.scrollTop)) + 'px';
          gutter.appendChild(led);
        }
      });
    }

    function hoverLine(e){
      const rect = urlInput.getBoundingClientRect();
      const y = e.clientY - rect.top + urlInput.scrollTop;
      const lh = parseFloat(getComputedStyle(urlInput).lineHeight) || 22;
      const padTop = parseFloat(getComputedStyle(urlInput).paddingTop) || 12;
      const idx = Math.max(0, Math.round((y - padTop - lh/2)/lh));
      hover.style.top = (padTop + idx*lh - urlInput.scrollTop) + 'px';
      hover.style.opacity = '1';
    }
    function leave(){ hover.style.opacity = '0'; }

    urlInput.addEventListener('input', refresh);
    urlInput.addEventListener('scroll', refresh, { passive:true });
    if (isDesktop) {
      urlInput.addEventListener('mousemove', hoverLine, { passive:true });
      urlInput.addEventListener('mouseleave', leave);
    }
    refresh();
  })();

  /* ---------- [7] Reroll “Shuffle Merge” ---------- */
  (function hookRerollAnimation(){
    if (typeof window.updateTweetBlock !== 'function') return;
    const _updateTweetBlock = window.updateTweetBlock;
    window.updateTweetBlock = function animatedUpdate(tweetEl, result) {
      try {
        // animate existing lines out
        const oldLines = $$('.comment-line', tweetEl);
        oldLines.forEach(l => l.classList.add('ct-out'));
        // wait briefly, then perform original DOM swap
        setTimeout(() => {
          _updateTweetBlock(tweetEl, result);
          // animate new lines in
          $$('.comment-line', tweetEl).forEach(l => l.classList.add('ct-in'));
          setTimeout(() => $$('.comment-line', tweetEl).forEach(l => l.classList.remove('ct-in')), 700);
        }, Math.min(220, Math.max(80, oldLines.length*30)));
      } catch (e) {
        // fallback to original if anything odd
        _updateTweetBlock(tweetEl, result);
      }
    };
  })();

  /* ---------- [13] Tab Title “done” on cancellation too ---------- */
  // when cancel button toggles state off, MutationObserver above handles it,
  // but also catch explicit Cancel UI if used without class flip (safety).
  const cancelBtn = document.getElementById('cancelBtn');
  cancelBtn && cancelBtn.addEventListener('click', () => setTitleDone());

  /* ---------- Friendly hooks ---------- */
  // announce warmups/other small events if you want:
  // window.ctToast('Warmup pinged', 'ok');

})();

/* =========================
   CrownTALK — Emergency Desktop Safe Mode + Yielded Render
   Paste at the VERY END of your main JS file.
========================= */
(function () {
  const isDesktop = matchMedia('(pointer:fine)').matches;

  // A. Global “kill switch” for animations/observers if we see a long task on desktop
  let tripped = false;
  if (isDesktop && 'PerformanceObserver' in window) {
    try {
      const po = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          if (e.duration > 120 && !tripped) {
            tripped = true;
            document.documentElement.classList.add('ct-safe');
            console.warn('[CrownTALK] SAFE MODE enabled (long task detected):', e);
          }
        }
      });
      po.observe({ entryTypes: ['longtask'] });
    } catch {}
  }

  // B. Manual toggle: Shift+Esc to toggle SAFE mode
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && e.shiftKey) {
      document.documentElement.classList.toggle('ct-safe');
      console.warn('[CrownTALK] SAFE MODE toggled:', document.documentElement.classList.contains('ct-safe'));
    }
  });

  // C. Guard: cap skeletons + yield during rendering large batches
  // Monkey-patch showSkeletons if it exists
  if (typeof window.showSkeletons === 'function') {
    const _showSkeletons = window.showSkeletons;
    window.showSkeletons = function (count) {
      const capped = Math.min(Math.max(count, 1), 8); // cap to 8 for desktop stability
      return _showSkeletons(capped);
    };
  }

  // D. Yield helper for render loops: prevents main-thread stalls on desktop
  async function yieldNow() {
    if ('scheduler' in window && scheduler.yield) return scheduler.yield();
    await new Promise((r) => requestAnimationFrame(() => r()));
  }

  // E. Wrap your appendResultBlock to yield every few items (if present)
  if (typeof window.appendResultBlock === 'function') {
    const _append = window.appendResultBlock;
    let renderCount = 0;
    window.appendResultBlock = async function (result) {
      // Every 3rd card, yield a frame on desktop
      if (isDesktop) {
        renderCount++;
        if (renderCount % 3 === 0) await yieldNow();
      }
      return _append(result);
    };
  }

  // F. Ensure Generate flow never totally blocks: if your handleGenerate uses setTimeout
  //    bursts, we add a defensive progress heartbeat so UI stays alive.
  if (!window.__ctHeartbeat) {
    window.__ctHeartbeat = setInterval(() => {
      const el = document.getElementById('progressBarFill');
      if (!el) return;
      // tiny nudge to keep a repaint happening while busy
      if (document.body.classList.contains('is-generating')) {
        const w = parseFloat(el.style.transform?.match(/scaleX\(([^)]+)/)?.[1] || '0');
        const nudged = Math.min(0.99, w + 0.002);
        el.style.transform = `scaleX(${nudged})`;
      }
    }, 500);
  }

  // G. Disable expensive observers/HUD if present when in SAFE mode
  const mo = new MutationObserver(() => {
    if (document.documentElement.classList.contains('ct-safe')) {
      // Hide HUD, confetti, ambient threads if they exist
      const kill = (sel) => document.querySelectorAll(sel).forEach((n) => (n.style.display = 'none'));
      kill('#ctHud, .ct-confetti, .ambient-thread');
    }
  });
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

})();
