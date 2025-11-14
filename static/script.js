// =========================================
// CrownTALK v2.0 — Frontend Brain
// =========================================

// DOM elements
const input = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const resultsContainer = document.getElementById("resultsContainer");
const statusArea = document.getElementById("statusArea");

// Backend endpoint on Koyeb
const API_URL = "/comment";


// ================================
// Copy-to-Clipboard Helper
// ================================
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



// ================================
// Render Each Tweet Result
// ================================
function renderTweetResult(index, url, data) {
    const box = document.createElement("div");
    box.className = "result-box fadeIn";

    const header = document.createElement("div");
    header.className = "tweet-header";
    header.innerHTML = `<strong>${index}. </strong><a href="${url}" target="_blank">${url}</a>`;
    box.appendChild(header);

    // Error from backend
    if (data.error) {
        const err = document.createElement("div");
        err.className = "tweet-error";
        err.textContent = `⚠️ ${data.error}`;
        box.appendChild(err);
        resultsContainer.appendChild(box);
        return;
    }

    // Two comments
    data.comments.forEach((comment) => {
        const line = document.createElement("div");
        line.className = "comment-line";

        const textSpan = document.createElement("span");
        textSpan.textContent = comment;

        const btn = document.createElement("button");
        btn.className = "copy-btn";
        btn.textContent = "Copy";

        btn.addEventListener("click", () => copyToClipboard(comment, btn));

        line.appendChild(textSpan);
        line.appendChild(btn);
        box.appendChild(line);
    });

    resultsContainer.appendChild(box);
}



// ================================
// Batch Processor (2 per batch)
// ================================
async function processInBatches(tweetLinks) {
    resultsContainer.innerHTML = "";
    statusArea.textContent = "";
    let batchNum = 1;

    const batches = [];
    for (let i = 0; i < tweetLinks.length; i += 2) {
        batches.push(tweetLinks.slice(i, i + 2));
    }

    for (const batch of batches) {
        statusArea.textContent = `Processing batch ${batchNum} of ${batches.length} — please wait...`;

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tweets: batch })
            });

            const data = await response.json();

            if (data.results) {
                batch.forEach((url, idx) => {
                    renderTweetResult(
                        (batchNum - 1) * 2 + (idx + 1),
                        url,
                        data.results[idx]
                    );
                });
            }
        } catch (e) {
            batch.forEach((url, idx) => {
                renderTweetResult(
                    (batchNum - 1) * 2 + (idx + 1),
                    url,
                    { error: "The comment generator is temporarily unavailable." }
                );
            });
        }

        // Wait 10–12 seconds before next batch
        if (batchNum < batches.length) {
            statusArea.textContent = `Waiting 10–12 seconds before next batch…`;
            await new Promise((res) => setTimeout(res, 10000 + Math.random() * 2000));
        }

        batchNum++;
    }

    statusArea.innerHTML = `✔ All comments generated successfully.`;
}



// ================================
// Handle Generate Button Click
// ================================
generateBtn.addEventListener("click", async () => {
    const raw = input.value.trim();

    if (!raw) {
        alert("Please paste at least one X link.");
        return;
    }

    const links = raw.split("\n").map((l) => l.trim()).filter(Boolean);

    if (links.length === 0) {
        alert("No valid links found.");
        return;
    }

    generateBtn.disabled = true;
    generateBtn.textContent = "Processing...";

    await processInBatches(links);

    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Replies";
});
