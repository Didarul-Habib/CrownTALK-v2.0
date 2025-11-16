const backendBase = "https://crowntalk-v2-0.onrender.com";
const commentURL = `${backendBase}/comment`;
const rerollURL = `${backendBase}/reroll`;

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

// year in footer
if (yearEl) {
  yearEl.textContent = new Date().getFullYear();
}

// ---------- UI theme (background / glow) ----------

const themeDots = document.querySelectorAll(".theme-dot");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("crowntalk_theme", theme);
  themeDots.forEach(dot => {
    dot.classList.toggle("active", dot.dataset.theme === theme);
  });
}

const storedTheme = localStorage.getItem("crowntalk_theme") || "dark-purple";
applyTheme(storedTheme);

themeDots.forEach(dot => {
  dot.addEventListener("click", () => {
    applyTheme(dot.dataset.theme);
  });
});

// ---------- Comment theme (tone) ----------

const commentThemeButtons = document.querySelectorAll(".comment-theme-btn");
let currentCommentTheme =
  localStorage.getItem("crowntalk_comment_theme") || "default";

function applyCommentTheme(theme) {
  currentCommentTheme = theme;
  commentThemeButtons.forEach(btn => {
    btn.classList.toggle(
      "active",
      btn.dataset.commentTheme === theme
    );
  });
  localStorage.setItem("crowntalk_comment_theme", theme);
}

// init
if (commentThemeButtons.length) {
  // ensure data-comment-theme attr name is readable from JS
  commentThemeButtons.forEach(btn => {
    if (!btn.dataset.commentTheme && btn.dataset.commenttheme) {
      btn.dataset.commentTheme = btn.dataset.commenttheme;
    }
  });
  applyCommentTheme(currentCommentTheme);
  commentThemeButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      applyCommentTheme(btn.dataset.commentTheme);
    });
  });
}

// ---------- Helpers ----------

let currentController = null;
let cancelled = false;
let clipboardHistory = [];

function parseUrls(raw) {
  return raw
    .split(/\n+/)
    .map(line => line.trim())
    .filter(Boolean);
}

function clearOutputs() {
  resultsEl.innerHTML = "";
  failedEl.innerHTML = "";
  progressEl.textContent = "";
  progressBarFill.style.width = "0%";
  updateCounts(0, 0);
}

function updateCounts(totalResults, totalFailed) {
  resultCountEl.textContent = `${totalResults} tweet${
    totalResults === 1 ? "" : "s"
  }`;
  failedCountEl.textContent = `${totalFailed}`;
}

function setGenerating(isGenerating) {
  generateBtn.disabled = isGenerating;
  cancelBtn.disabled = !isGenerating;
  document.body.classList.toggle("is-generating", isGenerating);
  generateBtn.textContent = isGenerating ? "Working..." : "Generate Comments";
}

// split "native (english)" comment into { native, english }
function splitComment(comment) {
  if (!comment || typeof comment !== "string") {
    return { native: "", english: null };
  }
  const match = comment.match(/^(.*)\(([^)]*)\)\s*$/);
  if (match) {
    const native = match[1].trim();
    const english = match[2].trim();
    if (native && english) {
      return { native, english };
    }
  }
  return { native: comment.trim(), english: null };
}

function truncate(text, maxLen) {
  if (!text || text.length <= maxLen) return text || "";
  return text.slice(0, maxLen - 1).trimEnd() + "…";
}

// normalize comments from different backend formats
function normalizeCommentsFromItem(item) {
  const out = [];

  // new style: item.comment = { en: "...", bn/hi/xx: "..." }
  if (item.comment && typeof item.comment === "object") {
    const c = item.comment;
    const en = c.en || c.EN || null;

    // possible native keys (extend if needed)
    const langKeys = [
      "bn", "hi", "ur", "ar", "es", "pt", "id", "tr",
      "ph", "ta", "te", "ml", "fr", "de", "it", "ru", "zh"
    ];

    let native = null;
    for (const key of langKeys) {
      if (typeof c[key] === "string" && c[key].trim()) {
        native = c[key].trim();
        break;
      }
    }

    if (!native && en) native = en;

    if (native || en) {
      out.push({ native: native || "", english: en });
    }
  }

  // old style: item.comments = ["native (english)", ...]
  if (Array.isArray(item.comments)) {
    item.comments.forEach(comment => {
      if (typeof comment === "string") {
        out.push(splitComment(comment));
      } else if (comment && typeof comment === "object") {
        // if backend someday sends { native, english }
        const native = comment.native || comment.text || "";
        const english = comment.english || null;
        out.push(splitComment(english ? `${native} (${english})` : native));
      }
    });
  }

  return out;
}

// extract meta info safely
function getMeta(item) {
  const meta = item.meta || {};
  const themeLabel =
    meta.theme_label || meta.themeLabel || meta.theme || null;
  const moodLabel =
    meta.mood_label || meta.moodLabel || meta.mood || null;
  const langLabel =
    meta.lang_label || meta.langLabel || meta.lang || null;
  const preview =
    meta.preview || meta.excerpt || meta.tweet_preview || null;

  return { themeLabel, moodLabel, langLabel, preview };
}

// ---------- Main flow ----------

generateBtn.addEventListener("click", async () => {
  const raw = urlInput.value.trim();
  if (!raw) return;

  const urls = parseUrls(raw);
  if (urls.length === 0) return;

  clearOutputs();
  progressEl.textContent = `Processing ${urls.length} URLs...`;
  progressBarFill.style.width = "5%";

  cancelled = false;
  currentController = new AbortController();
  setGenerating(true);

  try {
    const payload = {
      urls,
      theme: currentCommentTheme   // backend can use or ignore
    };

    const res = await fetch(commentURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: currentController.signal
    });

    if (!res.ok) {
      if (!cancelled) {
        progressEl.textContent = "Backend error while processing.";
        progressBarFill.style.width = "100%";
      }
      return;
    }

    const data = await res.json();
    if (cancelled) return;

    const batches = (data && data.batches) || [];

    let totalResults = 0;
    let totalFailed = 0;
    const totalBatches = Math.max(batches.length, 1);

    let delay = 50;

    batches.forEach((batch, idx) => {
      if (cancelled) return;
      const batchIndex = batch.batch || idx + 1;
      const batchResults = batch.results || [];
      const batchFailed = batch.failed || [];

      totalResults += batchResults.length;
      totalFailed += batchFailed.length;

      const progressChunk = ((idx + 1) / totalBatches) * 100;
      progressEl.textContent = `Batch ${batchIndex} done (${idx + 1}/${totalBatches})`;
      progressBarFill.style.width = `${progressChunk}%`;

      batchResults.forEach(item => {
        if (cancelled) return;

        const skeletonBlock = createSkeletonBlock(item.url);
        resultsEl.appendChild(skeletonBlock);

        setTimeout(() => {
          if (!cancelled) {
            fillResultBlock(skeletonBlock, item);
            updateCounts(totalResults, totalFailed);
          }
        }, delay);

        delay += 120;
      });

      batchFailed.forEach(item => {
        if (cancelled) return;
        renderFailed(item);
      });
    });

    if (batches.length === 0) {
      progressEl.textContent = "No valid URLs found.";
      progressBarFill.style.width = "100%";
    } else {
      setTimeout(() => {
        if (!cancelled) {
          progressEl.textContent = "All batches completed!";
          progressBarFill.style.width = "100%";
          updateCounts(totalResults, totalFailed);
        }
      }, delay + 150);
    }
  } catch (err) {
    if (cancelled) {
      progressEl.textContent = "Generation cancelled.";
      progressBarFill.style.width = "0%";
    } else {
      console.error(err);
      progressEl.textContent = "Network error while contacting backend.";
      progressBarFill.style.width = "100%";
    }
  } finally {
    setGenerating(false);
    currentController = null;
  }
});

cancelBtn.addEventListener("click", () => {
  if (!currentController) return;
  cancelled = true;
  currentController.abort();
  setGenerating(false);
  progressEl.textContent = "Generation cancelled.";
  progressBarFill.style.width = "0%";
});

clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearOutputs();
});

// ---------- Rendering helpers ----------

function createSkeletonBlock(url) {
  const block = document.createElement("div");
  block.className = "result-block skeleton";

  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url;

  const skeletonWrapper = document.createElement("div");
  skeletonWrapper.className = "skeleton-bars";

  const bar1 = document.createElement("div");
  bar1.className = "skeleton-bar";
  const bar2 = document.createElement("div");
  bar2.className = "skeleton-bar";

  const typing = document.createElement("div");
  typing.className = "typing-indicator";
  typing.innerHTML = `
      generating<span class="typing-dots">
        <span></span><span></span><span></span>
      </span>
  `;

  skeletonWrapper.appendChild(bar1);
  skeletonWrapper.appendChild(bar2);

  block.appendChild(link);
  block.appendChild(skeletonWrapper);
  block.appendChild(typing);

  return block;
}

function fillResultBlock(block, item) {
  block.classList.remove("skeleton");
  block.innerHTML = "";

  // header row with link + single reroll button
  const headerRow = document.createElement("div");
  headerRow.className = "comment-header-row";

  const link = document.createElement("a");
  link.href = item.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = item.url;

  const rerollBtn = document.createElement("button");
  rerollBtn.className = "reroll-btn";
  rerollBtn.textContent = "re-roll";
  rerollBtn.dataset.url = item.url;

  headerRow.appendChild(link);
  headerRow.appendChild(rerollBtn);
  block.appendChild(headerRow);

  // meta chips + tweet preview (if backend provides)
  const { themeLabel, moodLabel, langLabel, preview } = getMeta(item);

  if (themeLabel || moodLabel || langLabel) {
    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";

    if (themeLabel) {
      const chip = document.createElement("span");
      chip.className = "meta-chip theme-chip";
      chip.textContent = `Theme: ${themeLabel}`;
      metaRow.appendChild(chip);
    }

    if (moodLabel) {
      const chip = document.createElement("span");
      chip.className = "meta-chip mood-chip";
      chip.textContent = `Mood: ${moodLabel}`;
      metaRow.appendChild(chip);
    }

    if (langLabel) {
      const chip = document.createElement("span");
      chip.className = "meta-chip lang-chip";
      chip.textContent = `Lang: ${langLabel}`;
      metaRow.appendChild(chip);
    }

    block.appendChild(metaRow);
  }

  if (preview) {
    const previewEl = document.createElement("div");
    previewEl.className = "tweet-preview";
    previewEl.textContent = truncate(preview, 160);
    block.appendChild(previewEl);
  }

  // comments (multi-lang + EN copy)
  const comments = normalizeCommentsFromItem(item);
  const toRender = comments.length > 0 ? comments : [{ native: "No comment", english: null }];

  toRender.forEach(({ native, english }) => {
    const line = document.createElement("div");
    line.className = "comment-line";

    const span = document.createElement("span");
    span.className = "comment-text";
    span.textContent = english && english !== native
      ? `${native} (${english})`
      : native;

    const buttonGroup = document.createElement("div");
    buttonGroup.style.display = "flex";
    buttonGroup.style.gap = "6px";

    // native copy
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "copy";
    copyBtn.dataset.text = native || english || "";
    buttonGroup.appendChild(copyBtn);

    // english copy if available
    if (english && english !== native) {
      const copyEnBtn = document.createElement("button");
      copyEnBtn.className = "copy-btn copy-btn-en";
      copyEnBtn.textContent = "EN";
      copyEnBtn.dataset.text = english;
      buttonGroup.appendChild(copyEnBtn);
    }

    line.appendChild(span);
    line.appendChild(buttonGroup);
    block.appendChild(line);
  });

  // trigger highlight animation (initial + reroll)
  block.classList.remove("flash-highlight");
  // force reflow so animation can restart
  void block.offsetWidth;
  block.classList.add("flash-highlight");
}

function renderFailed(item) {
  const div = document.createElement("div");
  div.className = "failed-entry";
  div.textContent = `Failed: ${item.url} — ${item.reason}`;
  failedEl.appendChild(div);
}

// ---------- Clipboard history ----------

function pushHistory(text) {
  if (!text) return;
  clipboardHistory.unshift(text);
  clipboardHistory = clipboardHistory.slice(0, 20);
  renderHistory();
}

function renderHistory() {
  historyEl.innerHTML = "";
  if (clipboardHistory.length === 0) {
    const empty = document.createElement("div");
    empty.className = "history-entry";
    empty.innerHTML = "<span class='history-text'>No copied comments yet.</span>";
    historyEl.appendChild(empty);
    return;
  }

  clipboardHistory.forEach(txt => {
    const entry = document.createElement("div");
    entry.className = "history-entry";

    const span = document.createElement("span");
    span.className = "history-text";
    span.textContent = txt;

    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.dataset.text = txt;

    entry.appendChild(span);
    entry.appendChild(btn);
    historyEl.appendChild(entry);
  });
}

clearHistoryBtn.addEventListener("click", () => {
  clipboardHistory = [];
  renderHistory();
});

renderHistory();

// ---------- Global click handler: copy + reroll ----------

document.addEventListener("click", async e => {
  const target = e.target;

  // copy (native or EN or history)
  if (target.classList.contains("copy-btn")) {
    const text = target.dataset.text || "";
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      pushHistory(text);

      const old = target.textContent;
      target.textContent = "copied";
      target.disabled = true;
      setTimeout(() => {
        target.textContent = old;
        target.disabled = false;
      }, 900);
    } catch (err) {
      console.error("Copy failed", err);
    }
    return;
  }

  // per-tweet re-roll (single button per tweet)
  if (target.classList.contains("reroll-btn")) {
    const url = target.dataset.url;
    if (!url) return;

    const block = target.closest(".result-block");
    if (!block) return;

    target.disabled = true;
    const oldLabel = target.textContent;
    target.textContent = "re-rolling...";

    const tempIndicator = document.createElement("div");
    tempIndicator.className = "typing-indicator";
    tempIndicator.innerHTML =
      `refreshing<span class="typing-dots"><span></span><span></span><span></span></span>`;
    block.appendChild(tempIndicator);

    try {
      const res = await fetch(rerollURL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, theme: currentCommentTheme })
      });
      const data = await res.json();

      if (data && !data.error) {
        fillResultBlock(block, {
          url: data.url || url,
          comments: data.comments,
          comment: data.comment,
          meta: data.meta
        });
      } else {
        console.error("Reroll failed", data && data.error);
      }
    } catch (err) {
      console.error("Reroll network error", err);
    } finally {
      target.disabled = false;
      target.textContent = oldLabel;
      tempIndicator.remove();
    }
  }
});
