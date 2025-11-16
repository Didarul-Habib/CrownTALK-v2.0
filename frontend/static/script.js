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
      progressEl.textContent = "Backend error while process
