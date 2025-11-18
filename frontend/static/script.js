// ------------------------
// Backend endpoints
// ------------------------
const backendBase = "https://crowntalk.onrender.com";
const commentURL = `${backendBase}/comment`;
const rerollURL = `${backendBase}/reroll`;

// ------------------------
// DOM elements
// ------------------------
const urlInput = document.getElementById("urlInput");
const generateBtn = document.getElementById("generateBtn");
const cancelBtn = document.getElementById("cancelBtn");
const clearBtn = document.getElementById("clearBtn");

const progressEl = document.getElementById("progress");
const progressBarFill = document.getElementById("progressBarFill");

const resultsEl = document.getElementById("results");
const failedEl = document.getElementById("failed");
const resultCountEl = document.getElementById("resultCount");
const failedCountEl = document.getElementById("failedCount");

const historyEl = document.getElementById("history");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const yearEl = document.getElementById("year");

// theme dots
const themeDots = Array.from(document.querySelectorAll(".theme-dot"));

// ------------------------
// State
// ------------------------
let cancelled = false;
let historyItems = [];

// ------------------------
// Utilities
// ------------------------
function parseURLs(raw) {
  if (!raw) return [];
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      // remove "1. https://..."
      line = line.replace(/^\s*\d+\.\s*/, "");
      return line.trim();
    });
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
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  } catch (err) {
    console.error("Clipboard error", err);
  }
}

function formatTweetCount(count) {
  const n = Number(count) || 0;
  return `${n} tweet${n === 1 ? "" : "s"}`;
}

// Textarea auto-grow
function autoResizeTextarea() {
  if (!urlInput) return;
  urlInput.style.height = "auto";
  const base = 180; // matches CSS min-height baseline
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

  const items = [...historyItems].reverse();

  items.forEach((item) => {
    const entry = document.createElement("div");
    // Align with CSS: use .history-item (previously .history-entry)
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
      <svg
        width="12"
        height="12"
        fill="#0E418F"
        xmlns="http://www.w3.org/2000/svg"
        shape-rendering="geometricPrecision"
        text-rendering="geometricPrecision"
        image-rendering="optimizeQuality"
        fill-rule="evenodd"
        clip-rule="evenodd"
        viewBox="0 0 467 512.22"
      >
        <path
          fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z"
        ></path>
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

  // header
  const header = document.createElement("div");
  header.className = "tweet-header";

  const link = document.createElement("a");
  link.className = "tweet-link";
  link.href = url || "#";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url || "(no url)";

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
    if (comment.lang) {
      line.dataset.lang = comment.lang;
    }

    const tag = document.createElement("span");
    tag.className = "comment-tag";
    if (multilingual) {
      tag.textContent =
        comment.lang === "en" ? "EN" : (comment.lang || "native").toUpperCase();
    } else {
      tag.textContent = `EN #${idx + 1}`;
    }

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
      <svg
        width="12"
        height="12"
        fill="#0E418F"
        xmlns="http://www.w3.org/2000/svg"
        shape-rendering="geometricPrecision"
        text-rendering="geometricPrecision"
        image-rendering="optimizeQuality"
        fill-rule="evenodd"
        clip-rule="evenodd"
        viewBox="0 0 467 512.22"
      >
        <path
          fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z"
        ></path>
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
  }); // <-- important: close forEach properly

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

  // highlight animation after reroll
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

// Skeletons while waiting
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

  cancelled = false;
  document.body.classList.add("is-generating");

  generateBtn.disabled = true;
  cancelBtn.disabled = false;
  resetResults();
  resetProgress();

  setProgressText(`Processing ${urls.length} URL${urls.length === 1 ? "" : "s"}…`);
  setProgressRatio(0.03);
  showSkeletons(urls.length);

  try {
    const res = await fetch(commentURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
    });

    if (!res.ok) {
      throw new Error(`Backend error: ${res.status}`);
    }

    const data = await res.json();
    if (cancelled) return;

    const results = Array.isArray(data.results) ? data.results : [];
    const failed = Array.isArray(data.failed) ? data.failed : [];

    resultsEl.innerHTML = "";
    failedEl.innerHTML = "";

    let processed = 0;
    const total = results.length || urls.length;

    // show results one by one (not all at once)
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
          cancelBtn.disabled = true;
        }
      }, delay);
      delay += 120;
    });

    // failed URLs
    failed.forEach((f) => appendFailedItem(f));
    failedCountEl.textContent = String(failed.length);

    // if no results at all
    if (!results.length) {
      document.body.classList.remove("is-generating");
      generateBtn.disabled = false;
      cancelBtn.disabled = true;
      if (failed.length) {
        setProgressText("All URLs failed to process.");
      } else {
        setProgressText("No comments returned.");
      }
      setProgressRatio(1);
    }
  } catch (err) {
    console.error("Generate error", err);
    document.body.classList.remove("is-generating");
    generateBtn.disabled = false;
    cancelBtn.disabled = true;
    setProgressText("Error contacting CrownTALK backend.");
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
  cancelBtn.disabled = true;
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
  const url = tweetEl && tweetEl.dataset.url;
  if (!url) return;

  const button = tweetEl.querySelector(".reroll-btn");
  if (!button) return;

  const oldLabel = button.textContent;
  button.disabled = true;
  button.textContent = "Rerolling…";

  const commentsWrap = tweetEl.querySelector(".comments");
  if (commentsWrap) {
    commentsWrap.innerHTML = "";
    const sk1 = document.createElement("div");
    sk1.className = "tweet-skeleton-line";
    const sk2 = document.createElement("div");
    sk2.className = "tweet-skeleton-line";
    commentsWrap.appendChild(sk1);
    commentsWrap.appendChild(sk2);
  }

  try {
    const res = await fetch(rerollURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      throw new Error(`Reroll failed: ${res.status}`);
    }
    const data = await res.json();
    if (data && Array.isArray(data.comments)) {
      updateTweetBlock(tweetEl, {
        url: data.url || url,
        comments: data.comments,
      });
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

function applyTheme(themeName) {
  const html = document.documentElement;
  html.setAttribute("data-theme", themeName);

  themeDots.forEach((dot) => {
    if (dot.dataset.theme === themeName) {
      dot.classList.add("is-active");
    } else {
      dot.classList.remove("is-active");
    }
  });

  try {
    localStorage.setItem(THEME_STORAGE_KEY, themeName);
  } catch {
    // ignore
  }
}

function initTheme() {
  let theme = "white";
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) theme = stored;
  } catch {
    // ignore
  }
  applyTheme(theme);
}

// ------------------------
// Event bindings
// ------------------------
document.addEventListener("DOMContentLoaded", () => {
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }

  initTheme();
  renderHistory();
  autoResizeTextarea();

  if (urlInput) {
    urlInput.addEventListener("input", autoResizeTextarea);
  }

  generateBtn.addEventListener("click", () => {
    if (document.body.classList.contains("is-generating")) return;
    handleGenerate();
  });

  cancelBtn.addEventListener("click", handleCancel);
  clearBtn.addEventListener("click", handleClear);

  clearHistoryBtn.addEventListener("click", () => {
    historyItems = [];
    renderHistory();
  });

  // Copy / reroll in results
  resultsEl.addEventListener("click", async (event) => {
    const copyBtn = event.target.closest(".copy-btn, .copy-btn-en");
    const rerollBtn = event.target.closest(".reroll-btn");

    if (copyBtn) {
      const text = copyBtn.dataset.text || "";
      if (!text) return;

      await copyToClipboard(text);
      addToHistory(text);

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
      if (!tweetEl) return;
      handleReroll(tweetEl);
    }
  });

  // History copy
  historyEl.addEventListener("click", async (event) => {
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

  // theme dots
  themeDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const t = dot.dataset.theme;
      if (!t) return;
      applyTheme(t);
    });
  });
});



/* ===== CrownTALK: minimal add-on (default theme = dark-purple) =====
   Drop this at the very bottom of static/script.js.
   It safely redefines initTheme to prefer 'dark-purple' on first load.
*/

function initTheme() {
  let theme = "dark-purple"; // default for first-time visitors
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) theme = stored; // respect user’s previous choice
  } catch {
    // ignore storage errors
  }
  applyTheme(theme);
}

