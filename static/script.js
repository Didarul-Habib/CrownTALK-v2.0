// -------------------------------------------------------------
// CrownTALK Frontend Engine — Premium Edition
// -------------------------------------------------------------

const inputBox = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const batchStatus = document.getElementById("batchStatus");
const resultsContainer = document.getElementById("resultsContainer");
const backToTop = document.getElementById("backToTop");

// -------------------------------------------------------------
// BACK TO TOP BUTTON
// -------------------------------------------------------------
window.addEventListener("scroll", () => {
    backToTop.style.display = window.scrollY > 600 ? "block" : "none";
});

backToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// -------------------------------------------------------------
// CLEAN & NORMALIZE URL
// -------------------------------------------------------------
function cleanURL(url) {
    if (!url) return "";
    url = url.trim();

    url = url.replace("mobile.twitter.com", "x.com");
    url = url.replace("twitter.com", "x.com");
    url = url.replace("vxtwitter.com", "x.com");
    url = url.replace("fxtwitter.com", "x.com");

    if (!url.startsWith("http")) {
        url = "https://" + url;
    }
    return url.split("?")[0];
}

// -------------------------------------------------------------
// COPY COMMENT
// -------------------------------------------------------------
function copyComment(element, text) {
    navigator.clipboard.writeText(text).then(() => {
        element.classList.add("copied");
        setTimeout(() => element.classList.remove("copied"), 800);
    });
}

// -------------------------------------------------------------
// RENDER RESULT CARD
// -------------------------------------------------------------
function renderResultCard(url, comments, error = null) {
    const card = document.createElement("div");
    card.className = "result-card";

    if (error) {
        card.innerHTML = `
            <span class="tweet-link">${url}</span>
            <div style="color:#ff9696; font-size:0.95rem;">⚠️ ${error}</div>
        `;
        resultsContainer.appendChild(card);
        return;
    }

    let commentHTML = "";
    comments.forEach((c) => {
        commentHTML += `
            <div class="comment-row">
                <span class="comment-text">${c}</span>
                <div class="copy-btn" title="Copy">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="#f7d67c">
                        <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14
                                 c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 
                                 16H8V7h11v14z"/>
                    </svg>
                </div>
            </div>
        `;
    });

    card.innerHTML = `
        <span class="tweet-link">${url}</span>
        ${commentHTML}
    `;

    // Attach copy events
    card.querySelectorAll(".copy-btn").forEach((btn, index) => {
        btn.addEventListener("click", () => {
            copyComment(btn, comments[index]);
        });
    });

    resultsContainer.appendChild(card);

    // Auto-scroll
    card.scrollIntoView({ behavior: "smooth", block: "center" });
}

// -------------------------------------------------------------
// BATCH PROCESSING
// -------------------------------------------------------------
async function sendBatch(batch, batchIndex, totalBatches) {
    batchStatus.textContent =
        `⚙️ Processing batch ${batchIndex} of ${totalBatches} — please wait...`;

    try {
        const response = await fetch("/comment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweets: batch }),
        });

        const data = await response.json();

        if (!data.results) {
            batch.forEach((url) =>
                renderResultCard(url, null, "The generator returned an invalid response")
            );
            return;
        }

        data.results.forEach((res, i) => {
            const originalURL = batch[i];

            if (res.error) {
                renderResultCard(originalURL, null, res.error);
            } else {
                renderResultCard(originalURL, res.comments, null);
            }
        });

    } catch (err) {
        batch.forEach((url) =>
            renderResultCard(url, null, "Server unreachable — try again shortly")
        );
    }
}

// -------------------------------------------------------------
// MAIN GENERATE FUNCTION
// -------------------------------------------------------------
generateBtn.addEventListener("click", async () => {
    const raw = inputBox.value.trim();
    if (!raw) return;

    // Prevent spamming
    generateBtn.disabled = true;
    generateBtn.textContent = "Processing...";

    resultsContainer.innerHTML = "";
    batchStatus.textContent = "";

    // Split + clean URLs
    let urls = raw.split("\n")
                  .map((u) => cleanURL(u))
                  .filter((u) => u.length > 5);

    if (urls.length === 0) {
        batchStatus.textContent = "⚠️ No valid tweet links found.";
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Replies";
        return;
    }

    // Create batches of 2
    let batches = [];
    for (let i = 0; i < urls.length; i += 2) {
        batches.push(urls.slice(i, i + 2));
    }

    const totalBatches = batches.length;

    for (let i = 0; i < totalBatches; i++) {
        await sendBatch(batches[i], i + 1, totalBatches);

        // Delay between batches (Koyeb safe)
        if (i < totalBatches - 1) {
            batchStatus.textContent = `⏳ Waiting 10 seconds before next batch...`;
            await new Promise((resolve) => setTimeout(resolve, 10000));
        }
    }

    batchStatus.textContent = "✅ All comments generated successfully.";

    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Replies";
});
