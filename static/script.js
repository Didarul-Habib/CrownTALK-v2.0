// CrownTALK v2 — Frontend Logic
// Handles batching, progress, copy buttons, animations

const textarea = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const outputArea = document.getElementById("output");
const statusLine = document.getElementById("statusLine");


// ----------------------------------------------------------
// Helper: Sleep for batching (10–12 sec random)
// ----------------------------------------------------------
function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// ----------------------------------------------------------
// Copy button handler
// ----------------------------------------------------------
function attachCopyHandlers() {
    document.querySelectorAll(".copy-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const text = btn.getAttribute("data-copy");
            navigator.clipboard.writeText(text);
            btn.innerText = "Copied!";
            btn.classList.add("copied");

            setTimeout(() => {
                btn.innerText = "Copy";
                btn.classList.remove("copied");
            }, 1500);
        });
    });
}


// ----------------------------------------------------------
// MAIN: Generate Comments
// ----------------------------------------------------------
generateBtn.addEventListener("click", async () => {

    let urls = textarea.value
        .trim()
        .split("\n")
        .map(u => u.trim())
        .filter(u => u.length > 0);

    if (urls.length === 0) {
        alert("Please paste at least one tweet link.");
        return;
    }

    outputArea.innerHTML = "";
    statusLine.innerHTML = "";
    generateBtn.disabled = true;
    generateBtn.innerText = "Processing...";


    // --------------------------
    // Batching Logic (2 at a time)
    // --------------------------
    let batchSize = 2;
    let totalBatches = Math.ceil(urls.length / batchSize);

    for (let i = 0; i < urls.length; i += batchSize) {
        let batchNum = Math.floor(i / batchSize) + 1;
        let batch = urls.slice(i, i + batchSize);

        statusLine.innerHTML = `Processing batch ${batchNum} of ${totalBatches}…`;

        // send to backend
        try {
            let r = await fetch("/comment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tweets: batch })
            });

            let data = await r.json();

            data.results.forEach((result, idx) => {
                let url = batch[idx];

                if (result.error) {
                    outputArea.innerHTML += `
                        <div class="tweet-block error-block fadeIn">
                            <div class="tweet-url">🔗 ${url}</div>
                            <div class="error-text">⚠️ ${result.error}</div>
                        </div>
                    `;
                } else {
                    let c1 = result.comments[0];
                    let c2 = result.comments[1];

                    outputArea.innerHTML += `
                        <div class="tweet-block fadeIn">
                            <div class="tweet-url">🔗 ${url}</div>

                            <div class="comment-line">
                                ${c1}
                                <button class="copy-btn" data-copy="${c1}">Copy</button>
                            </div>

                            <div class="comment-line">
                                ${c2}
                                <button class="copy-btn" data-copy="${c2}">Copy</button>
                            </div>

                            <div class="divider"></div>
                        </div>
                    `;
                }
            });

            attachCopyHandlers();

        } catch (err) {
            outputArea.innerHTML += `
                <div class="tweet-block error-block fadeIn">
                    <div class="error-text">⚠️ The comment generator is temporarily unavailable.</div>
                </div>
            `;
        }

        // wait before next batch
        if (batchNum < totalBatches) {
            statusLine.innerHTML = `Waiting 10–12 seconds before next batch...`;
            await wait(10000 + Math.random() * 2000);
        }
    }

    // finished!
    statusLine.innerHTML = "✅ All comments generated successfully.";
    generateBtn.disabled = false;
    generateBtn.innerText = "Generate Replies";
});
