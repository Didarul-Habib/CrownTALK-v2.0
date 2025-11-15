// ====== CONFIG ======
const BACKEND = "https://crowntalk-v2-0.onrender.com";

// ====== Helpers ======
function extractUrls(inputText) {
  const lines = inputText.split("\n");
  const urls = [];
  for (let line of lines) {
    if (!line.trim()) continue;
    // Strip numbering like: "1.", "2)", "3 -", "10:", "#11 ", etc.
    line = line.replace(/^\s*[\#\s]*\d+[\.\)\-:]*\s*/, "").trim();
    // Grab first URL on the line
    const m = line.match(/https?:\/\/[^\s]+/);
    if (m) {
      let url = m[0].trim();
      // Drop trailing punctuation like ")" or "."
      url = url.replace(/[),.]+$/, "");
      urls.push(url);
    }
  }
  // De‑dupe while preserving order
  return [...new Set(urls)];
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[c]));
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    if (btn) {
      const old = btn.textContent;
      btn.textContent = "Copied!";
      btn.disabled = true;
      setTimeout(() => {
        btn.textContent = old;
        btn.disabled = false;
      }, 800);
    }
  });
}

// ====== UI references ======
const $input = document.getElementById("inputUrls");
const $btn = document.getElementById("submitBtn");
const $out = document.getElementById("output");

// ====== Main click handler ======
$btn.addEventListener("click", async () => {
  const raw = $input.value.trim();
  $out.innerHTML = "";

  if (!raw) {
    $out.innerHTML = `<p style="color:#b91c1c">Please paste at least one URL.</p>`;
    return;
  }

  const urls = extractUrls(raw);
  if (urls.length === 0) {
    $out.innerHTML = `<p style="color:#b91c1c">No valid URLs found.</p>`;
    return;
  }

  // Progress header
  const header = document.createElement("div");
  header.innerHTML = `<p style="margin:8px 0;color:#555">Processing ${urls.length} link(s) in batches of 2…</p>`;
  $out.appendChild(header);

  // Containers
  const resultsBox = document.createElement("div");
  const failedBox = document.createElement("div");
  failedBox.innerHTML = `<h3 style="margin-top:16px;">Failed</h3>`;
  let failedCount = 0;

  $out.appendChild(resultsBox);
  $out.appendChild(failedBox);

  // Disable button during run
  $btn.disabled = true;
  $btn.textContent = "Working…";

  // Process in batches of 2 and append as we go
  for (let i = 0; i < urls.length; i += 2) {
    const batch = urls.slice(i, i + 2);

    // Show batch progress
    header.innerHTML = `<p style="margin:8px 0;color:#555">Batch ${Math.floor(i / 2) + 1} of ${Math.ceil(urls.length / 2)}…</p>`;

    try {
      const res = await fetch(`${BACKEND}/comment`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ urls: batch })
      });

      const data = await res.json();

      // Render results from this batch
      if (Array.isArray(data.results)) {
        data.results.forEach(item => {
          const url = item.url || "";
          const c0 = (item.comments && item.comments[0]) ? item.comments[0] : "";
          const c1 = (item.comments && item.comments[1]) ? item.comments[1] : "";

          const block = document.createElement("div");
          block.className = "tweet-block";
          block.style.cssText = "border:1px solid #ddd;padding:10px;border-radius:6px;margin-top:10px;";

          block.innerHTML = `
            <p style="margin:0 0 6px;"><a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a></p>

            <div class="comment-line" style="display:flex;gap:8px;align-items:center;margin:4px 0;">
              <span>${escapeHtml(c0)}</span>
              <button class="copyBtn" style="padding:3px 8px;border:1px solid #bbb;background:#eee;border-radius:4px;cursor:pointer;">Copy</button>
            </div>

            <div class="comment-line" style="display:flex;gap:8px;align-items:center;margin:4px 0;">
              <span>${escapeHtml(c1)}</span>
              <button class="copyBtn" style="padding:3px 8px;border:1px solid #bbb;background:#eee;border-radius:4px;cursor:pointer;">Copy</button>
            </div>
          `;

          // Wire copy buttons
          const btns = block.querySelectorAll(".copyBtn");
          if (btns[0]) btns[0].addEventListener("click", () => copyText(c0, btns[0]));
          if (btns[1]) btns[1].addEventListener("click", () => copyText(c1, btns[1]));

          resultsBox.appendChild(block);
        });
      }

      // Render failed from this batch
      if (Array.isArray(data.failed) && data.failed.length > 0) {
        data.failed.forEach(item => {
          // supports either plain string URLs or {url, reason}
          let url = "";
          let reason = "Unknown reason";
          if (typeof item === "string") {
            url = item;
          } else if (item && typeof item === "object") {
            url = item.url || "";
            reason = item.reason || reason;
          }
          const row = document.createElement("p");
          row.style.margin = "4px 0";
          row.innerHTML = `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url) || "(empty URL)"}</a> — <span style="color:#b91c1c">${escapeHtml(reason)}</span>`;
          failedBox.appendChild(row);
          failedCount++;
        });
      }
    } catch (err) {
      const row = document.createElement("p");
      row.style.color = "#b91c1c";
      row.textContent = `Batch ${Math.floor(i/2)+1} failed: network/server error`;
      failedBox.appendChild(row);
      failedCount++;
      // continue with next batch
    }

    // Gentle delay to avoid hitting rate limits too hard
    await sleep(500);
  }

  header.innerHTML = `<p style="margin:8px 0;color:#0f766e">Done. ${failedCount ? failedCount + " failed." : "All succeeded."}</p>`;
  $btn.disabled = false;
  $btn.textContent = "Generate Comments";
});
