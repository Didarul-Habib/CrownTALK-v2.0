const backendURL = "https://crowntalk-v2-0.onrender.com/comment";

const urlInput = document.getElementById("urlInput");
const generateBtn = document.getElementById("generateBtn");
const clearBtn = document.getElementById("clearBtn");
const progressEl = document.getElementById("progress");
const resultsEl = document.getElementById("results");
const failedEl = document.getElementById("failed");

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
}

generateBtn.addEventListener("click", async () => {
  const raw = urlInput.value.trim();
  if (!raw) return;

  const urls = parseUrls(raw);
  if (urls.length === 0) return;

  clearOutputs();
  progressEl.textContent = `Processing ${urls.length} URLs...`;

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
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate Comments";
      return;
    }

    const data = await res.json();
    const batches = data.batches || [];

    batches.forEach((batch, idx) => {
      progressEl.textContent = `Batch ${batch.batch} done (${idx + 1}/${batches.length})`;

      (batch.results || []).forEach(renderResult);
      (batch.failed || []).forEach(renderFailed);
    });

    progressEl.textContent = "All batches completed!";
  } catch (err) {
    console.error(err);
    progressEl.textContent = "Network error while contacting backend.";
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Comments";
  }
});

clearBtn.addEventListener("click", () => {
  urlInput.value = "";
  clearOutputs();
});

function renderResult(item) {
  const block = document.createElement("div");
  block.className = "result-block";

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

  resultsEl.appendChild(block);
}

function renderFailed(item) {
  const div = document.createElement("div");
  div.className = "failed-entry";
  div.textContent = `Failed: ${item.url} — ${item.reason}`;
  failedEl.appendChild(div);
}

// Global click handler for copy buttons
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
    }, 800);
  });
});
