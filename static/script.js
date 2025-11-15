// =====================
// GLOBAL STATE
// =====================
let stopRequested = false;

// =====================
// THEME SYSTEM
// =====================
const themeSelect = document.getElementById("themeSelect");
const themeToggle = document.getElementById("themeToggle");

themeSelect.addEventListener("change", () => {
    document.body.className = themeSelect.value;
});

themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light");
});

// =====================
// CLEAN URL
// =====================
function cleanUrl(url) {
    return url.split("?")[0].trim();
}

// =====================
// CREATE RESULT BLOCK IN UI
// =====================
function createResultBlock(url, index) {
    return `
        <div class="result-block">
            <div class="result-url">
                <strong>${index}.</strong>
                <a href="${url}" target="_blank">${url}</a>
            </div>
            <div class="comments"></div>
        </div>
    `;
}

// =====================
// COPY BUTTON
// =====================
function copyText(text) {
    navigator.clipboard.writeText(text);
}

// =====================
// GENERATE COMMENTS
// =====================
async function generateBatch(batchLinks, batchIndex, totalBatches) {
    if (stopRequested) return;

    document.getElementById("statusArea").innerHTML =
        `⚙️ Processing batch ${batchIndex} / ${totalBatches}...`;

    const response = await fetch("/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tweets: batchLinks }),
    });

    const data = await response.json();
    return data.results;
}

// =====================
// MAIN PROCESS
// =====================
document.getElementById("generateBtn").addEventListener("click", async () => {
    stopRequested = false;
    document.getElementById("stopBtn").classList.remove("hidden");
    document.getElementById("generateBtn").classList.add("hidden");

    const rawLinks = document.getElementById("tweetInput").value.trim().split("\n");
    const links = rawLinks.map(cleanUrl).filter(x => x.length > 5);
    if (links.length === 0) return;

    const batchSize = 2;
    const totalBatches = Math.ceil(links.length / batchSize);

    document.getElementById("results").innerHTML = "";

    let indexCounter = 1;

    for (let i = 0; i < links.length; i += batchSize) {
        if (stopRequested) break;

        const batch = links.slice(i, i + batchSize);
        const results = await generateBatch(batch, (i / batchSize) + 1, totalBatches);

        results.forEach((res, idx) => {
            const url = batch[idx];
            document.getElementById("results").innerHTML += createResultBlock(url, indexCounter);

            const block = document.querySelectorAll(".result-block")[indexCounter - 1].querySelector(".comments");

            if (res.error) {
                block.innerHTML = `<div class="comment-line">${res.error}</div>`;
            } else {
                res.comments.forEach(c => {
                    block.innerHTML += `
                        <div class="comment-line">
                            ${c}
                            <button class="copy-btn" onclick="copyText('${c}')">Copy</button>
                        </div>
                    `;
                });
            }
            indexCounter++;
        });

        await new Promise(r => setTimeout(r, 400)); // faster but safe rate
    }

    document.getElementById("statusArea").innerHTML =
        stopRequested ? "❌ Stopped." : "✔ All comments generated successfully.";

    document.getElementById("generateBtn").classList.remove("hidden");
    document.getElementById("stopBtn").classList.add("hidden");
});

// =====================
// STOP BUTTON
// =====================
document.getElementById("stopBtn").addEventListener("click", () => {
    stopRequested = true;
});
