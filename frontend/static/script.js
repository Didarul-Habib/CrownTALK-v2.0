// frontend/static/script.js

// ------------------------
// Backend endpoints
// ------------------------
const backendBase = "https://crowntalk-v2-0.onrender.com";
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

// Ensure progress bar has correct class for CSS animation
if (progressBarFill) {
  progressBarFill.classList.add("progress-bar-fill");
}

// ------------------------
// Utility helpers
// ------------------------
function parseURLs(raw) {
  if (!raw) return [];
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      // remove leading numbering like "1. https://..."
      line = line.replace(/^\s*\d+\.\s*/, "");
      return line.trim();
    });
}

function setProgressText(text) {
  progressEl.textContent = text || "";
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
  resultsEl.innerHTML = "";
  failedEl.innerHTML = "";
  resultCountEl.textContent = "0 tweets";
  failedCountEl.textContent = "0";
}

async function copyToClipboard(text) {
  if (!text) return;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      // fallback
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

// ------------------------
// History rendering
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
  historyEl.innerHTML = "";
  if (!historyItems.length) {
    historyEl.textContent = "Copied comments will show up here.";
    return;
  }

  // Newest last or newest first? Let’s do newest first.
  const items = [...historyItems].reverse();

  items.forEach((item) => {
    const entry = document.createElement("div");
    entry.className = "history-entry";

    const textSpan = document.createElement("div");
    textSpan.className = "history-text";
    textSpan.textContent = item.text;

    const side = document.createElement("div");
    side.style.display = "flex";
    side.style.flexDirection = "column";
    side.style.alignItems = "flex-end";
    side.style.gap = "4px";

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = item.timestamp;

    const btn = document.createElement("button");
    btn.className = "history-copy-btn";
    btn.textContent = "Copy";
    btn.dataset.text = item.text;

    side.appendChild(meta);
    side.appendChild(btn);

    entry.appendChild(textSpan);
    entry.appendChild(side);
    historyEl.appendChild(entry);
  });
}

// ------------------------
// Result rendering
// ------------------------
function buildTweetBlock(result) {
  const url = result.url || "";
  const comments = Array.isArray(result.comments) ? result.comments : [];

  const tweet = document.createElement("div");
  tweet.className = "tweet";
  tweet.dataset.url = url;

  // Header
  const header = document.createElement("div");
  header.className = "tweet-header";

  const link = document.createElement("a");
  link.className = "tweet-link";
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url;

  const actions = document.createElement("div");
  actions.className = "tweet-actions";

  const rerollBtn = document.createElement("button");
  rerollBtn.className = "reroll-btn";
  rerollBtn.textContent = "Reroll";

  actions.appendChild(rerollBtn);
  header.appendChild(link);
  header.appendChild(actions);
  tweet.appendChild(header);

  // Comments
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
        comment.lang === "en"
          ? "EN"
          : (comment.lang || "native").toUpperCase();
    } else {
      tag.textContent = `EN #${idx + 1}`;
    }

    const bubble = document.createElement("span");
    bubble.className = "comment-text";
    bubble.textContent = comment.text;

    const copyBtn = document.createElement("button");
    if (multilingual) {
      if (comment.lang === "en") {
        copyBtn.className = "copy-btn-en";
        copyBtn.textContent = "Copy EN";
      } else {
        copyBtn.className = "copy-btn";
        copyBtn.textContent = "Copy";
      }
    } else {
      copyBtn.className = "copy-btn";
      copyBtn.textContent = "Copy";
    }
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

// Update an existing block during reroll
function fillResultBlock(tweetEl, result) {
  if (!tweetEl) return;
  tweetEl.dataset.url = result.url || tweetEl.dataset.url || "";

  // Replace comments section
  const oldComments = tweetEl.querySelector(".comments");
  if (oldComments) oldComments.remove();

  const replacement = buildTweetBlock(result).querySelector(".comments");
  if (replacement) tweetEl.appendChild(replacement);

  // Highlight animation inline (no CSS changes needed)
  tweetEl.style.transition = "box-shadow 0.25s ease, transform 0.25s ease";
  const oldBoxShadow = tweetEl.style.boxShadow;
  const oldTransform = tweetEl.style.transform;
  tweetEl.style.boxShadow = "0 0 0 2px rgba(56,189,248,0.9)";
  tweetEl.style.transform = "translateY(-1px)";
  setTimeout(() => {
    tweetEl.style.boxShadow = oldBoxShadow;
    tweetEl.style.transform = oldTransform;
  }, 420);
}

// Failed item
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

// Skeletons for initial loading
function showSkeletons(count) {
  resultsEl.innerHTML = "";
  const num = Math.min(Math.max(count, 1), 6); // cap skeletons so it doesn't get insane
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

    // Flexible response handling
    let batches = [];
    if (data && Array.isArray(data.batches)) {
      batches = data.batches;
    } else {
      batches = [
        {
          batch: 1,
          results: (data && data.results) || [],
          failed: (data && data.failed) || [],
        },
      ];
    }

    let totalResults = 0;
    let totalFailed = 0;
    const totalBatches = Math.max(batches.length, 1);

    // Swap out skeletons
    resultsEl.innerHTML = "";

    let delay = 60;

    batches.forEach((batch, idx) => {
      if (cancelled) return;
      const batchIndex = batch.batch || idx + 1;
      const batchResults = Array.isArray(batch.results) ? batch.results : [];
      const batchFailed = Array.isArray(batch.failed) ? batch.failed : [];

      setTimeout(() => {
        if (cancelled) return;

        batchResults.forEach((result) => {
          appendResultBlock(result);
          totalResults += 1;
        });

        batchFailed.forEach((failure) => {
          appendFailedItem(failure);
          totalFailed += 1;
        });

        const processedBatches = idx + 1;
        const ratio = processedBatches / totalBatches;
        setProgressRatio(ratio);

        if (totalBatches > 1) {
          if (processedBatches < totalBatches) {
            setProgressText(
              `Batch ${processedBatches} done, batch ${
                processedBatches + 1
              } running…`
            );
          } else {
            setProgressText(
              `All ${totalBatches} batch${
                totalBatches === 1 ? "" : "es"
              } completed.`
            );
          }
        } else {
          setProgressText(
            `Processed ${totalResults + totalFailed} URL${
              totalResults + totalFailed === 1 ? "" : "s"
            }.`
          );
        }

        resultCountEl.textContent = formatTweetCount(totalResults);
        failedCountEl.textContent = String(totalFailed);

        if (processedBatches === totalBatches) {
          document.body.classList.remove("is-generating");
          generateBtn.disabled = false;
          cancelBtn.disabled = true;
        }
      }, delay);

      delay += 180;
    });
  } catch (err) {
    console.error("Generate error", err);
    document.body.classList.remove("is-generating");
    generateBtn.disabled = false;
    cancelBtn.disabled = true;
    setProgressText("Error contacting CrownTALK backend.");
  }
}

// ------------------------
// Cancel + Clear
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
}

// ------------------------
// Theme handling
// ------------------------
const THEME_STORAGE_KEY = "crowntalk_theme";

function applyTheme(themeName) {
  const html = document.documentElement;
  html.setAttribute("data-theme", themeName);

  // active class update
  themeDots.forEach((dot) => {
    if (dot.dataset.theme === themeName) {
      dot.classList.add("is-active");
    } else {
      dot.classList.remove("is-active");
    }
  });

  try {
    localStorage.setItem(THEME_STORAGE_KEY, themeName);
  } catch (_) {
    // ignore
  }
}

function initTheme() {
  let theme = "blue";
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) theme = stored;
  } catch (_) {
    // ignore
  }
  applyTheme(theme);
}

// ------------------------
// Event bindings
// ------------------------
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

// Copy buttons inside results
resultsEl.addEventListener("click", async (event) => {
  const copyBtn = event.target.closest(".copy-btn, .copy-btn-en");
  const rerollBtn = event.target.closest(".reroll-btn");

  if (copyBtn) {
    const text = copyBtn.dataset.text || "";
    if (!text) return;

    await copyToClipboard(text);
    addToHistory(text);

    const oldLabel = copyBtn.textContent;
    copyBtn.textContent = "Copied";
    copyBtn.disabled = true;
    setTimeout(() => {
      copyBtn.textContent = oldLabel;
      copyBtn.disabled = false;
    }, 700);
  }

  if (rerollBtn) {
    const tweetEl = rerollBtn.closest(".tweet");
    const url = tweetEl && tweetEl.dataset.url;
    if (!url) return;

    const oldLabel = rerollBtn.textContent;
    rerollBtn.disabled = true;
    rerollBtn.textContent = "Rerolling…";

    // transient skeleton inside tweet for reroll
    const comments = tweetEl.querySelector(".comments");
    if (comments) {
      comments.innerHTML = "";
      const sk1 = document.createElement("div");
      sk1.className = "tweet-skeleton-line";
      const sk2 = document.createElement("div");
      sk2.className = "tweet-skeleton-line";
      comments.appendChild(sk1);
      comments.appendChild(sk2);
    }

    (async () => {
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

        if (data && !data.error && Array.isArray(data.comments)) {
          fillResultBlock(tweetEl, {
            url: data.url || url,
            comments: data.comments,
          });
        } else {
          console.error("Reroll backend error", data && data.error);
          setProgressText("Reroll failed for this tweet.");
        }
      } catch (err) {
        console.error("Reroll network error", err);
        setProgressText("Network error during reroll.");
      } finally {
        rerollBtn.disabled = false;
        rerollBtn.textContent = oldLabel;
      }
    })();
  }
});

// History copy buttons
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

// Theme dots
themeDots.forEach((dot) => {
  dot.addEventListener("click", () => {
    const t = dot.dataset.theme;
    if (!t) return;
    applyTheme(t);
  });
});

// ------------------------
// Init
// ------------------------
document.addEventListener("DOMContentLoaded", () => {
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }
  initTheme();
  renderHistory();
});
