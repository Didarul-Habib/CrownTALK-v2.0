// ------------------------------------------------------
// CrownTALK vFinal - Premium Frontend Logic
// ------------------------------------------------------

const generateBtn = document.getElementById("generateBtn");
const urlInput = document.getElementById("urlInput");
const progressBox = document.getElementById("progressBox");
const resultsBox = document.getElementById("results");
const copyAllBtn = document.getElementById("copyAllBtn");

// Helper: Sleep
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Typing animation effect
function typeEffect(text) {
    progressBox.innerText = "";
    let i = 0;

    let typer = setInterval(() => {
        progressBox.innerText = text.substring(0, i);
        i++;
        if (i > text.length) clearInterval(typer);
    }, 18);
}

// Main event
generateBtn.addEventListener("click", async () => {
    const rawInput = urlInput.value.trim();

    if (!rawInput) {
        alert("Paste at least one tweet link.");
        return;
    }

    // Clean and split URLs
    const urls = rawInput
        .split("\n")
        .map(u => u.trim())
        .filter(u => u.length > 0);

    if (urls.length === 0) {
        alert("No valid tweet links found.");
        return;
    }

    // Reset UI
    resultsBox.innerHTML = "";
    progressBox.classList.remove("hidden");
    typeEffect("Starting CrownTALK engine…");
    copyAllBtn.classList.add("hidden");

    try {
        await sleep(400);

        typeEffect("Processing tweets… preparing batches…");

        const response = await fetch("/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls })
        });

        const data = await response.json();

        progressBox.classList.add("hidden");

        // Build combined text for copy-all
        let finalText = "";

        data.results.forEach((item, i) => {
            const card = document.createElement("div");
            card.className =
                "resultCard fadeUpAnim cardLift";

            if (item.error) {
                card.innerHTML = `
                    <p class="text-yellow-400 font-bold goldUnderline">
                        🔗 ${item.url}
                    </p>
                    <p class="mt-3 text-red-400">${item.error}</p>
                `;
            } else {
                card.innerHTML = `
                    <p class="text-yellow-400 font-bold goldUnderline">
                        🔗 ${item.url}
                    </p>
                    <p class="mt-3">${item.comment1}</p>
                    <p class="mt-1">${item.comment2}</p>
                `;

                finalText += `${item.comment1}\n${item.comment2}\n\n`;
            }

            resultsBox.appendChild(card);
        });

        // Enable copy-all
        copyAllBtn.classList.remove("hidden");

        copyAllBtn.onclick = () => {
            navigator.clipboard.writeText(finalText);
            copyAllBtn.innerText = "Copied!";
            copyAllBtn.classList.add("copyPulse");

            setTimeout(() => {
                copyAllBtn.classList.remove("copyPulse");
                copyAllBtn.innerText = "Copy All Comments";
            }, 1500);
        };

    } catch (err) {
        progressBox.classList.remove("hidden");
        progressBox.innerHTML = `<span class="text-red-400">⚠️ Something went wrong. Try again.</span>`;
    }
});
