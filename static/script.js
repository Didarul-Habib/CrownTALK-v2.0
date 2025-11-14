
/* ---------------------------------------------
   GLOBAL VARIABLES
--------------------------------------------- */

let stopGeneration = false;
let currentTheme = localStorage.getItem("crownTheme") || "purple";

const themeSwitcher = document.getElementById("themeSwitcher");
const tweetInput = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const stopBtn = document.getElementById("stopBtn");
const resultsContainer = document.getElementById("resultsContainer");
const retrySection = document.getElementById("retrySection");
const retryList = document.getElementById("retryList");
const statusMessage = document.getElementById("statusMessage");

/* ---------------------------------------------
   APPLY THEME
--------------------------------------------- */
function applyTheme(theme) {
    document.body.classList.remove(
        "theme-purple",
        "theme-midnight",
        "theme-gold",
        "theme-light"
    );
    document.body.classList.add(`theme-${theme}`);
    localStorage.setItem("crownTheme", theme);
}

applyTheme(currentTheme);
themeSwitcher.value = currentTheme;

/* ---------------- THEME SWITCHER ---------------- */

themeSwitcher.addEventListener("change", () => {
    currentTheme = themeSwitcher.value;
    applyTheme(currentTheme);
});

/* ---------------------------------------------
   CLEAN LINKS
--------------------------------------------- */
function cleanLink(url) {
    if (!url) return null;

    url = url.trim();

    if (!url.includes("twitter.com") && !url.includes("x.com")) {
        return null;
    }

    // Remove tracking parameters
    return url.split("?")[0];
}

/* ---------------------------------------------
   RETRY LOGIC FOR COMMENT GENERATION
--------------------------------------------- */
async function requestWithRetry(url) {
    const payload = JSON.stringify({ tweets: [url] });
    const headers = { "Content-Type": "application/json" };

    let attempt = 0;
    const delays = [1000, 2000, 4000]; // 1s, 2s, 4s

    while (attempt < 3) {
        try {
            const res = await fetch("/comment", {
                method: "POST",
                headers,
                body: payload,
            });

            const data = await res.json();
            return data.results[0];

        } catch (err) {
            await new Promise((r) => setTimeout(r, delays[attempt]));
            attempt++;
        }
    }

    return { error: "Failed after 3 retries" };
}

/* ---------------------------------------------
   CREATE RESULT CARD
--------------------------------------------- */
function createCard(url, data) {
    const card = document.createElement("div");
    card.className = "result-card";

    const safeUrl = url;

    card.innerHTML = `
        <a href="${safeUrl}" target="_blank" class="tweet-link">${safeUrl}</a>
    `;

    if (data.error) {
        card.innerHTML += `<p style="color:#ff5757;">⚠ ${data.error}</p>`;
        return card;
    }

    data.comments.forEach((comment) => {
        const line = document.createElement("div");
        line.className = "comment-line";

        line.innerHTML = `
            <span>${comment}</span>
            <button class="copy-btn">Copy</button>
        `;

        line.querySelector(".copy-btn").onclick = () => {
            navigator.clipboard.writeText(comment);
        };

        card.appendChild(line);
    });

    return card;
}

/* ---------------------------------------------
   STOP GENERATION
--------------------------------------------- */
stopBtn.addEventListener("click", () => {
    stopGeneration = true;
    statusMessage.textContent = "🛑 Stopped by user.";
});

/* ---------------------------------------------
   MAIN GENERATION FUNCTION
--------------------------------------------- */
generateBtn.addEventListener("click", async () => {

    stopGeneration = false;
    resultsContainer.innerHTML = "";
    retrySection.classList.add("hidden");
    retryList.innerHTML = "";
    statusMessage.textContent = "";

    generateBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");

    // Extract cleaned links
    let links = tweetInput.value
        .split("\n")
        .map(cleanLink)
        .filter(Boolean);

    links = [...new Set(links)];

    if (links.length === 0) {
        alert("No valid X/Twitter links found.");
        generateBtn.classList.remove("hidden");
        stopBtn.classList.add("hidden");
        return;
    }

    // Batch into groups of 2
    const batches = [];
    for (let i = 0; i < links.length; i += 2) {
        batches.push(links.slice(i, i + 2));
    }

    const failed = [];

    for (let b = 0; b < batches.length; b++) {
        if (stopGeneration) break;

        statusMessage.textContent = `⚙ Processing batch ${b + 1} / ${batches.length}...`;

        const batch = batches[b];

        for (const url of batch) {
            if (stopGeneration) break;

            const result = await requestWithRetry(url);

            if (result.error) {
                failed.push(url);
            }

            const card = createCard(url, result);
            resultsContainer.appendChild(card);
        }

        // Delay to avoid rate limit
        if (!stopGeneration) {
            await new Promise((r) => setTimeout(r, 11000));
        }
    }

    stopBtn.classList.add("hidden");
    generateBtn.classList.remove("hidden");

    // Show failed list
    if (failed.length > 0) {
        retrySection.classList.remove("hidden");
        statusMessage.textContent = "⚠ Some links failed.";

        failed.forEach((url) => {
            const li = document.createElement("li");
            li.textContent = url;
            retryList.appendChild(li);
        });
    } else {
        statusMessage.textContent = "✔ All comments generated.";
    }
});
