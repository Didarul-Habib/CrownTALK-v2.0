// Backend endpoints
const COMMENT_URL = "https://crowntalk-v2-0.onrender.com/comment";
const STREAM_URL  = "https://crowntalk-v2-0.onrender.com/comment_stream";

const $ = (s) => document.querySelector(s);
const output = $("#output");
const input  = $("#inputUrls");
const progressWrap = $("#progressWrap");
const progressBar  = $("#progressBar");
const progressText = $("#progressText");

let lastResults = [];

// Theme toggle
$("#toggleTheme").addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
});

// Clear
$("#clearBtn").addEventListener("click", () => {
  input.value = "";
  output.innerHTML = "";
  hideProgress();
  lastResults = [];
});

// Download JSON
$("#downloadBtn").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(lastResults, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "crowntalk_results.json";
  a.click();
  URL.revokeObjectURL(url);
});

// Classic (one-shot)
$("#classicBtn").addEventListener("click", async () => {
  const urls = parseUrls();
  if (!urls.length) return tip("No URLs provided.");

  resetUI(urls.length);
  setProgress(0, 1, "Submitting (classic)…");

  try {
    const res = await fetch(COMMENT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls })
    });
    if (!res.ok) throw new Error("server");

    const data = await res.json();
    showNoticeIfLocal(data?.results || []);
    (data?.results || []).forEach(renderResult);
    renderFailures(data?.failed || []);

    setProgress(1, 1, "Done");
  } catch (e) {
    tip("Connection error. Try again.", true);
  }
});

// Streaming (recommended)
$("#streamBtn").addEventListener("click", async () => {
  const urls = parseUrls();
  if (!urls.length) return tip("No URLs provided.");

  resetUI(urls.length);

  try {
    const res = await fetch(STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls })
    });
    if (!res.ok || !res.body) throw new Error("no-stream");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let totalBatches = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let lines = buffer.split("\n");
      buffer = lines.pop(); // keep partial

      for (const line of lines) {
        if (!line.trim()) continue;
        let msg;
        try { msg = JSON.parse(line); } catch { continue; }

        if (msg.type === "start") {
          totalBatches = msg.batches || 0;
          setProgress(0, totalBatches, "Starting…");
        }

        if (msg.type === "progress") {
          setProgress(msg.batch - 1, totalBatches, `Processing batch ${msg.batch}/${totalBatches}…`);
        }

        if (msg.type === "batch") {
          // Show results immediately
          showNoticeIfLocal(msg.results || []);
          for (const r of (msg.results || [])) renderResult(r);
          renderFailures(msg.failed || []);
          setProgress(msg.batch, totalBatches, `Batch ${msg.batch}/${totalBatches} complete`);
        }

        if (msg.type === "done") {
          setProgress(totalBatches, totalBatches, "All done");
        }
      }
    }
  } catch (e) {
    tip("Connection error. Try again.", true);
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// UI helpers
// ──────────────────────────────────────────────────────────────────────────────
function parseUrls() {
  return input.value
    .split("\n")
    .map((x) => x.trim())
    .filter(Boolean);
}

function tip(text, isError=false) {
  output.innerHTML = `<p class="${isError ? "bad" : ""}">${text}</p>`;
}

function resetUI(total) {
  output.innerHTML = `<div class="notice">Processing <b>${total}</b> URL(s)…</div>`;
  showProgress();
  setProgress(0, Math.ceil(total / 2), "Queued");
  lastResults = [];
}

function showProgress() { progressWrap.classList.remove("hidden"); }
function hideProgress() { progressWrap.classList.add("hidden"); }
function setProgress(doneBatches, totalBatches, label) {
  const pct = totalBatches ? Math.max(0, Math.min(100, Math.round((doneBatches / totalBatches) * 100))) : 0;
  progressBar.style.width = pct + "%";
  progressText.textContent = label || "";
}

function showNoticeIfLocal(results) {
  if (!results?.length) return;
  if (results.some(r => r.source === "offline")) {
    output.insertAdjacentHTML("afterbegin",
      `<div class="notice">OpenAI is rate-limited right now — switched to <b>Local</b> generator for some items.</div>`
    );
  }
}

function renderFailures(list) {
  if (!list?.length) return;
  // filter out non-per-URL items just in case
  const perUrl = list.filter(f => f.url && f.url !== "BATCH");
  if (!perUrl.length) return;

  let html = `<div class="block"><h3 class="bad">Failed</h3>`;
  for (const f of perUrl) {
    const msg = f.reason || "Unknown";
    html += `❌ <a href="${f.url}" target="_blank" rel="noopener">${f.url}</a> — ${msg}<br>`;
  }
  html += `</div>`;
  output.insertAdjacentHTML("beforeend", html);
}

function renderResult(r) {
  lastResults.push(r);
  const safe1 = r.comments?.[0] || "";
  const safe2 = r.comments?.[1] || "";
  const tag = r.source === "offline"
    ? '<span class="tag tag-local">Local</span>'
    : '<span class="tag tag-ai">AI</span>';

  const html = `
    <div class="block">
      <div class="row">
        <a href="${r.url}" target="_blank" rel="noopener">${r.url}</a> ${tag}
      </div>

      <div class="row">
        • ${escapeHtml(safe1)}
        <button class="copy-btn" onclick="copyText(\`${escapeTick(safe1)}\`)">Copy</button>
      </div>

      <div class="row">
        • ${escapeHtml(safe2)}
        <button class="copy-btn" onclick="copyText(\`${escapeTick(safe2)}\`)">Copy</button>
      </div>

      <div class="row">
        <button class="copy-btn" onclick="copyText(\`${escapeTick(safe1 + '\\n' + safe2)}\`)">Copy both</button>
        <a class="copy-btn" href="${r.url}" target="_blank" rel="noopener">Open tweet</a>
      </div>
    </div>
  `;
  output.insertAdjacentHTML("beforeend", html);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[m]));
}
function escapeTick(s) { return s.replace(/`/g, "\\`"); }

window.copyText = (t) => {
  navigator.clipboard.writeText(t).catch(()=>{});
};
