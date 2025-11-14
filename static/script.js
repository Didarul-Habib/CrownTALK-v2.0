// =========================================
// CrownTALK v2.5 — Neon Frontend Engine
// =========================================

// DOM
const input = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const resultsContainer = document.getElementById("resultsContainer");
const statusArea = document.getElementById("statusArea");

const API_URL = "/comment";


// Copy function
function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        button.textContent = "Copied!";
        button.classList.add("copied");

        setTimeout(() => {
            button.textContent = "Copy";
            button.classList.remove("copied");
        }, 1200);
    });
}


// Render tweet result
function renderTweetResult(index, url, data) {
    const box = document.createElement("div");
    box.className = "result-box";

    const header = document.createElement("div");
    header.className = "tweet-header";
    header.innerHTML = `<strong>${index}. </strong><a href="${url}" target="_blank">${url}</a>`;
    box.appendChild(header);

    if (data.error) {
        const err = document.createElement("div");
        err.style.color = "#ffb4b4";
        err.style.marginTop = "10px";
        err.textContent = `⚠️ ${data.error}`;
        box.appendChild(err);
        resultsContainer.appendChild(box);
        return;
    }

    data.comments.forEach((comment) => {
        const line = document.createElement("div");
        line.className = "comment-line";

        const span = document.createElement("span");
        span.textContent = comment;

        const btn = document.createElement("button");
        btn.className = "copy-btn";
        btn.textContent = "Copy";

        btn.addEventListener("click", () => copyToClipboard(comment, btn));

        line.appendChild(span);
        line.appendChild(btn);
        box.appendChild(line);
    });

    resultsContainer.appendChild(box);
}


// Batch processor
async function processInBatches(links) {
    resultsContainer.innerHTML = "";
    statusArea.textContent = "";

    const batches = [];
    for (let i = 0; i < links.length; i += 2) {
        batches.push(links.slice(i, i + 2));
    }

    let batchNum = 1;

    for (const batch of batches) {
        statusArea.textContent = `Processing batch ${batchNum} of ${batches.length}…`;

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tweets: batch })
            });

            const data = await response.json();

            batch.forEach((url, i) => {
                renderTweetResult(
                    (batchNum - 1) * 2 + (i + 1),
                    url,
                    data.results[i]
                );
            });

        } catch {
            batch.forEach((url, i) => {
                renderTweetResult(
                    (batchNum - 1) * 2 + (i + 1),
                    url,
                    { error: "The comment generator is temporarily unavailable." }
                );
            });
        }

        // Wait between batches
        if (batchNum < batches.length) {
            statusArea.textContent = `Waiting 10–12 seconds…`;
            await new Promise((res) => setTimeout(res, 10000 + Math.random() * 2000));
        }

        batchNum++;
    }

    statusArea.textContent = "✔ All comments generated successfully.";
}


// Button click
generateBtn.addEventListener("click", async () => {
    const raw = input.value.trim();

    if (!raw) {
        alert("Please paste at least one X link.");
        return;
    }

    const links = raw.split("\n").map((l) => l.trim()).filter(Boolean);

    generateBtn.disabled = true;
    generateBtn.textContent = "Processing…";

    await processInBatches(links);

    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Replies";
});
