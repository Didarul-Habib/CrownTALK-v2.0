// ------------------------------------------------------
// CrownTALK v2.0 — Gold Aura Engine
// Handles batching, API calls, UI rendering & copying
// ------------------------------------------------------

const generateBtn = document.getElementById("generateBtn");
const inputBox = document.getElementById("linksInput");
const resultsDiv = document.getElementById("results");
const statusBox = document.getElementById("statusBox");

const API_URL = "https://flask-twitter-api.onrender.com/comment";

// Delay helper
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Clean status
function setStatus(msg) {
    statusBox.textContent = msg;
}

// Clear UI
function clearResults() {
    resultsDiv.innerHTML = "";
}

// Copy button logic
function copyText(text, btn) {
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied!";
        btn.style.background = "rgba(0,255,120,0.35)";
        setTimeout(() => {
            btn.textContent = "Copy";
            btn.style.background = "rgba(255,200,60,0.2)";
        }, 1200);
    });
}

// Render a full tweet block
function renderResultBlock(url, comments) {
    const block = document.createElement("div");
    block.className = "result-block";

    const link = document.createElement("div");
    link.className = "result-url";
    link.textContent = url;

    block.appendChild(link);

    // If error returned
    if (typeof comments === "string") {
        const errorLine = document.createElement("div");
        errorLine.className = "comment-line";
        errorLine.textContent = comments;
        block.appendChild(errorLine);
        resultsDiv.appendChild(block);
        return;
    }

    // Render each comment
    comments.forEach((comment) => {
        const line = document.createElement("div");
        line.className = "comment-line";

        const textDiv = document.createElement("div");
        textDiv.textContent = comment;

        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-btn";
        copyBtn.textContent = "Copy";
        copyBtn.onclick = () => copyText(comment, copyBtn);

        line.appendChild(textDiv);
        line.appendChild(copyBtn);
        block.appendChild(line);
    });

    resultsDiv.appendChild(block);
}

// Fetch comments for 1–2 tweet URLs
async function processBatch(batchLinks, batchIndex, totalBatches) {
    setStatus(`Processing batch ${batchIndex} of ${totalBatches} — please wait…`);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweets: batchLinks }),
        });

        if (!response.ok) {
            batchLinks.forEach((url) => {
                renderResultBlock(url, "⚠️ The comment generator is temporarily unavailable. Try again shortly.");
            });
            return;
        }

        const data = await response.json();

        // Render results per tweet
        batchLinks.forEach((url, i) => {
            if (!data.results || !data.results[i]) {
                renderResultBlock(url, "⚠️ Could not fetch this tweet (private or deleted)");
                return;
            }

            const obj = data.results[i];

            if (obj.error) {
                renderResultBlock(url, `⚠️ ${obj.error}`);
            } else {
                renderResultBlock(url, obj.comments);
            }
        });

    } catch (err) {
        batchLinks.forEach((url) => {
            renderResultBlock(url, "⚠️ The comment generator is temporarily unavailable.");
        });
    }
}

generateBtn.addEventListener("click", async () => {
    clearResults();

    const raw = inputBox.value.trim();
    if (!raw) return setStatus("Please enter at least one tweet URL.");

    const links = raw
        .split("\n")
        .map((x) => x.trim())
        .filter((x) => x.length > 5);

    if (links.length === 0) {
        return setStatus("Please enter valid tweet URLs.");
    }

    setStatus("Starting…");

    // Split into batches of 2 (TweetAPI C requirement)
    let batches = [];
    for (let i = 0; i < links.length; i += 2) {
        batches.push(links.slice(i, i + 2));
    }

    const totalBatches = batches.length;

    for (let i = 0; i < totalBatches; i++) {
        await processBatch(batches[i], i + 1, totalBatches);

        if (i < totalBatches - 1) {
            setStatus("Waiting 10–12 seconds before next batch…");
            await sleep(11000);
        }
    }

    setStatus("✅ All comments generated successfully.");
});
