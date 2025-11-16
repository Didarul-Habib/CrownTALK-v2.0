const backendURL = "https://crowntalk-v2-0.onrender.com/comment";

const urlInput = document.getElementById("urlInput");
const generateBtn = document.getElementById("generateBtn");
const clearBtn = document.getElementById("clearBtn");
const progressEl = document.getElementById("progress");
const progressBarFill = document.getElementById("progressBarFill");
const resultsEl = document.getElementById("results");
const failedEl = document.getElementById("failed");
const resultCountEl = document.getElementById("resultCount");
const failedCountEl = document.getElementById("failedCount");
const yearEl = document.getElementById("year");

yearEl.textContent = new Date().getFullYear();

// -------- Theme handling --------

const themeDots = document.querySelectorAll(".theme-dot");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("crowntalk_theme", theme);
  themeDots.forEach(dot => {
    dot.classList.toggle("active", dot.dataset.theme === theme);
  });
}

const storedTheme = localStorage.getItem("crowntalk_theme") || "white";
applyTheme(storedTheme);

themeDots.forEach(dot => {
  dot.addEventListener("click", () => {
    applyTheme(dot.dataset.theme);
  });
});

// -------- Helpers --------

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
  resultCountEl.textContent = "0 tweets";
  failedCountEl.textContent = "0";
}

function updateCounts(totalResults, totalFailed) {
  resultCountEl.textContent = `${totalResults} tweet${totalResults === 1 ? "" : "s"}`;
  failedCountEl.textContent = `${totalFailed}`;
}

// -------- Main flow --------

generateBtn.addEventListener("click", async () => {
  const raw = urlInput.value.trim();
  if (!raw) return;

  const urls = parseUrls(raw);
  if (urls.length === 0) return;

  clearOutputs();
  progressEl.textContent = `Processing ${urls.length} URLs...`;
  progressBarFill.style.width = "5%";

  generateBtn.disabled = true;
  generateBtn.textContent = "Working...";

  try {
    const res = await fetch(backendURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls })
    });

    if (!res.ok) {
      progressEl.textContent = "Backend error while processing.";
      progressBarFill.style.width = "100%";
      return;
    }

    const data = await res.json();
    const batches = data.batches || [];

    let totalResults = 0;
    let totalFailed = 0;
    const totalBatches = Math.max(batches.length, 1);

    let delay = 50; // ms per tweet for fake streaming

    batches.forEach((batch, idx) => {
      const batchIndex = batch.batch || idx + 1;
      const batchResults = batch.results || [];
      const batchFailed = batch.failed || [];

      totalResults += batchResults.length;
      totalFailed += batchFailed.length;

      // progress bar chunk
      const progressChunk = ((idx + 1) / totalBatches) * 100;

      progressEl.textContent = `Batch ${batchIndex} done (${idx + 1}/${totalBatches})`;
      progressBarFill.style.width = `${progressChunk}%`;

      // skeleton + delayed fill for each result
      batchResults.forEach(item => {
        const skeletonBlock = createSkeletonBlock(item.url);
        resultsEl.appendChild(skeletonBlock);

        setTimeout(() => {
          fillResultBlock(skeletonBlock, item);
          updateCounts(totalResults, totalFailed);
        }, delay);

        delay += 120;
      });

      // failed ones appear immediately
      batchFailed.forEach(item => {
        renderFailed(item);
      });
    });

    if (batches.length === 0) {
      progressEl.textContent = "No valid URLs found.";
      progressBarFill.style.width = "100%";
    } else {
      setTimeout(() => {
        progressEl.textContent = "All batches completed!";
        progressBarFill.style.width = "100%";
        updateCounts(totalResults, totalFailed);
      }, delay + 150);
    }
  } catch (err) {
    console.error(err);
    progressEl.textContent = "Network error while contacting backend.";
    progressBarFill.style.width = "100%";
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Comments";
  }
});

clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearOutputs();
});

// -------- Rendering helpers --------

function createSkeletonBlock(url) {
  const block = document.createElement("div");
  block.className = "result-block skeleton";

  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url;

  const bar1 = document.createElement("div");
  bar1.className = "skeleton-bar";
  const bar2 = document.createElement("div");
  bar2.className = "skeleton-bar";

  block.appendChild(link);
  block.appendChild(bar1);
  block.appendChild(bar2);

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

    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.dataset.text = comment;

    line.appendChild(span);
    line.appendChild(btn);
    block.appendChild(line);
  });
}

function renderFailed(item) {
  const div = document.createElement("div");
  div.className = "failed-entry";
  div.textContent = `Failed: ${item.url} — ${item.reason}`;
  failedEl.appendChild(div);
}

// -------- Global copy handler --------

document.addEventListener("click", e => {
  if (!e.target.classList.contains("copy-btn")) return;

  const text = e.target.dataset.text || "";
  if (!text) return;

  navigator.clipboard.writeText(text).then(() => {
    const old = e.target.textContent;
    e.target.textContent = "copied";
    e.target.disabled = true;
    setTimeout(() => {
      e.target.textContent = old;
      e.target.disabled = false;
    }, 900);
  }).catch(err => {
    console.error("Copy failed", err);
  });
});
