const BACKEND = "https://crowntalk-v2-0.onrender.com";

const $ = (id) => document.getElementById(id);

$("submitBtn").addEventListener("click", async () => {
  const output = $("output");
  const btn = $("submitBtn");
  const raw = $("inputUrls").value.trim();

  if (!raw) {
    output.innerHTML = `<p class="error">Please enter at least one URL.</p>`;
    return;
  }

  const urls = raw.split("\n").map(u => u.trim()).filter(Boolean);
  btn.disabled = true;
  btn.textContent = "Working...";
  output.innerHTML = "<p>Processing…</p>";

  try {
    const res = await fetch(`${BACKEND}/comment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      output.innerHTML = `<p class="error">Server error: ${data.error || "Unknown error"}</p>`;
      btn.disabled = false;
      btn.textContent = "Generate Comments";
      return;
    }

    let html = "";
    if (Array.isArray(data.results)) {
      data.results.forEach(item => {
        html += `
          <div class="card">
            <h3><a href="${item.url}" target="_blank" rel="noopener">${item.url}</a></h3>
            <p><strong>Tweet:</strong> ${escapeHtml(item.tweet)}</p>
            <p><strong>Comment:</strong> ${escapeHtml(item.comment)}</p>
          </div>
        `;
      });
    }

    if (Array.isArray(data.failed) && data.failed.length > 0) {
      html += `<h2>Failed</h2>`;
      data.failed.forEach(f => {
        html += `<p class="failed"><a href="${f.url}" target="_blank" rel="noopener">${f.url}</a> — ${escapeHtml(f.reason || "Unknown reason")}</p>`;
      });
    }

    output.innerHTML = html || "<p>No results.</p>";
  } catch (e) {
    output.innerHTML = `<p class="error">Network or CORS error. Try again.</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Comments";
  }
});

function escapeHtml(s) {
  if (typeof s !== "string") return "";
  return s.replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
}
