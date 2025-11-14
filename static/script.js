let stopRequested = false;

const input = document.getElementById("tweetInput");
const generateBtn = document.getElementById("generateBtn");
const stopBtn = document.getElementById("stopBtn");
const resultsContainer = document.getElementById("results");
const batchStatus = document.getElementById("batchStatus");
const themeToggle = document.getElementById("themeToggle");

/* ---------------- CLEAN LINK FUNCTION ---------------- */
function cleanLink(url) {
    if (!url.includes("x.com") && !url.includes("twitter.com")) return null;
    return url.split("?")[0].trim();
}

/* ---------------- THEME TOGGLE ---------------- */
themeToggle.onclick = () => {
    document.body.classList.toggle("light-theme");
    document.body.classList.toggle("dark-theme");
};

/* ---------------- STOP GENERATION ---------------- */
stopBtn.onclick = () => {
    stopRequested = true;
    batchStatus.textContent = "🛑 Generation stopped.";
};

/* ---------------- MAIN PROCESS ---------------- */
generateBtn.onclick = async () => {
    stopRequested = false;
    resultsContainer.innerHTML = "";
    batchStatus.textContent = "";

    let links = input.value.split("\n").map(cleanLink).filter(Boolean);
    links = [...new Set(links)]; // remove duplicates

    if (links.length === 0) {
        alert("No valid tweet links found.");
        return;
    }

    generateBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");

    const batches = [];
    for (let i = 0; i < links.length; i += 2) {
        batches.push(links.slice(i, i + 2));
    }

    for (let i = 0; i < batches.length; i++) {
        if (stopRequested) break;

        batchStatus.textContent = `⚙ Processing batch ${i + 1} of ${batches.length} — please wait...`;

        const response = await fetch("/comment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweets: batches[i] }),
        });

        const data = await response.json();

        data.results.forEach((item, idx) => {
            let url = batches[i][idx];

            const card = document.createElement("div");
            card.className = "result-card";

            card.innerHTML = `
                <div class="tweet-url">${url}</div>
            `;

            if (item.error) {
                card.innerHTML += `<p style="color:#ff6262;">⚠ ${item.error}</p>`;
            } else {
                item.comments.forEach(comment => {
                    const line = document.createElement("div");
                    line.className = "comment-line";

                    line.innerHTML = `
                        <span>${comment}</span>
                        <button class="copy-btn">Copy</button>
                    `;

                    line.querySelector("button").onclick = () => {
                        navigator.clipboard.writeText(comment);
                    };

                    card.appendChild(line);
                });
            }

            resultsContainer.appendChild(card);
        });

        await new Promise(res => setTimeout(res, 10500));
    }

    stopBtn.classList.add("hidden");
    generateBtn.classList.remove("hidden");

    batchStatus.textContent = "✔ All comments generated successfully!";
};
