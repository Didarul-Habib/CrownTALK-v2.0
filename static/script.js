let stopRequested = false;

const inputField = document.getElementById("input");
const generateBtn = document.getElementById("generateBtn");
const stopBtn = document.getElementById("stopBtn");
const resultsContainer = document.getElementById("results");
const themeToggle = document.getElementById("themeToggle");

/* --------------------------------------------- */
/* Shorten long URLs for display */
/* --------------------------------------------- */

function shortenURL(url) {
    const clean = url.split("?")[0];
    if (clean.length <= 45) return clean;
    return clean.slice(0, 42) + "…";
}

/* --------------------------------------------- */
/* Theme Toggle */
/* --------------------------------------------- */

themeToggle.addEventListener("click", () => {
    document.documentElement.classList.toggle("light");
});

/* --------------------------------------------- */
/* Stop Button */
/* --------------------------------------------- */

stopBtn.addEventListener("click", () => {
    stopRequested = true;
    stopBtn.style.display = "none";
});

/* --------------------------------------------- */
/* Generate Replies */
/* --------------------------------------------- */

generateBtn.addEventListener("click", async () => {
    stopRequested = false;
    resultsContainer.innerHTML = "";

    let links = inputField.value
        .split("\n")
        .map(x => x.trim().split("?")[0])
        .filter(x => x.startsWith("http"));

    if (!links.length) return;

    generateBtn.style.display = "none";
    stopBtn.style.display = "block";

    const BATCH = 3;
    let batchNumber = 1;

    for (let i = 0; i < links.length; i += BATCH) {
        if (stopRequested) break;

        const batch = links.slice(i, i + BATCH);

        const status = document.createElement("div");
        status.style.margin = "10px 0";
        status.innerHTML = `⚙ Processing batch ${batchNumber} / ${Math.ceil(links.length / BATCH)}...`;
        resultsContainer.appendChild(status);

        const res = await fetch("/comment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweets: batch })
        });

        const data = await res.json();

        data.results.forEach((r, idx) => {
            const url = batch[idx];

            const item = document.createElement("div");
            item.className = "result-item";

            const link = document.createElement("a");
            link.className = "tweet-link";
            link.href = url;
            link.target = "_blank";
            link.textContent = shortenURL(url);

            item.appendChild(link);

            if (r.error) {
                const box = document.createElement("div");
                box.className = "comment-box";
                box.textContent = r.error;
                item.appendChild(box);
            } else {
                r.comments.forEach(c => {
                    const box = document.createElement("div");
                    box.className = "comment-box";
                    box.textContent = c;

                    const copy = document.createElement("div");
                    copy.className = "copy-btn";
                    copy.textContent = "Copy";
                    copy.onclick = () => {
                        navigator.clipboard.writeText(c);
                        copy.textContent = "Copied!";
                        setTimeout(() => (copy.textContent = "Copy"), 800);
                    };

                    box.appendChild(copy);
                    item.appendChild(box);
                });
            }

            resultsContainer.appendChild(item);
        });

        batchNumber++;

        if (i + BATCH < links.length) {
            await new Promise(r => setTimeout(r, 3500));
        }
    }

    stopBtn.style.display = "none";
    generateBtn.style.display = "block";
});

/* END */
