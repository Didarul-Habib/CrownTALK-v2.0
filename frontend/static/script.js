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

yearEl.textContent = new Date().getFullYear();

// ---------- Theme handling ----------

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
  resultCountEl.textContent = `${totalResults} tweet${totalResults === 1 ? "" : "s"}`;
  failedCountEl.textContent = `${totalFailed}`;
}

function setGenerating(isGenerating) {
  generateBtn.disabled = isGenerating;
  cancelBtn.disabled = !isGenerating;
  if (isGenerating) {
    generateBtn.textContent = "Working...";
  } else {
    generateBtn.textContent = "Generate Comments";
  }
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
    const res = await fetch(commentURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
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

    const batches = data.batches || [];

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

  const link = document.createElement("a");
  link.href = item.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = item.url;

  block.appendChild(link);

  (item.comments || []).forEach(comment => {
    const line = document.createElement("div");
    line.className = "comment-line";

    const span = document.createElement("span");
    span.className = "comment-text";
    span.textContent = comment;

    const buttonGroup = document.createElement("div");
    buttonGroup.style.display = "flex";
    buttonGroup.style.gap = "6px";

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "copy";
    copyBtn.dataset.text = comment;

    const rerollBtn = document.createElement("button");
    rerollBtn.className = "reroll-btn";
    rerollBtn.textContent = "re-roll";
    rerollBtn.dataset.url = item.url;

    buttonGroup.appendChild(copyBtn);
    buttonGroup.appendChild(rerollBtn);

    line.appendChild(span);
    line.appendChild(buttonGroup);

    block.appendChild(line);
  });
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
  // keep only last 20 items
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

  clipboardHistory.forEach((txt, idx) => {
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

// initial history state
renderHistory();

// ---------- Global click handler: copy + reroll ----------

document.addEventListener("click", async e => {
  const target = e.target;

  // copy
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

  // per-tweet re-roll
  if (target.classList.contains("reroll-btn")) {
    const url = target.dataset.url;
    if (!url) return;

    const block = target.closest(".result-block");
    if (!block) return;

    target.disabled = true;
    const oldLabel = target.textContent;
    target.textContent = "re-rolling...";

    // mini typing effect in this block
    const tempIndicator = document.createElement("div");
    tempIndicator.className = "typing-indicator";
    tempIndicator.innerHTML = `refreshing<span class="typing-dots"><span></span><span></span><span></span></span>`;
    block.appendChild(tempIndicator);

    try {
      const res = await fetch(rerollURL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const data = await res.json();

      if (data && !data.error && Array.isArray(data.comments)) {
        fillResultBlock(block, { url: data.url || url, comments: data.comments });
      } else {
        console.error("Reroll failed", data.error);
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
