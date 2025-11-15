//-------------------------------------------
// Extract clean URLs from textarea input
//-------------------------------------------
function extractUrls(inputText) {
    const lines = inputText.split("\n");
    const urls = [];

    for (let line of lines) {
        if (!line.trim()) continue;

        // Remove numbering such as: "1.", "2)", "3 -", "10:", etc.
        line = line.replace(/^\s*\d+[\.\)\-:]*\s*/, "").trim();

        // Extract actual URL
        const match = line.match(/https?:\/\/[^\s]+/);

        if (match) {
            let url = match[0].trim();

            // Clean trailing punctuation like ".", ",", ")", etc.
            url = url.replace(/[),.]+$/, "");

            urls.push(url);
        }
    }

    return urls;
}


//-------------------------------------------
// Handle "Generate Comments" button
//-------------------------------------------
document.getElementById("generateBtn").addEventListener("click", async function () {

    const rawInput = document.getElementById("urlsInput").value.trim();
    const resultsDiv = document.getElementById("results");
    const failedDiv = document.getElementById("failed");

    resultsDiv.innerHTML = "";
    failedDiv.innerHTML = "";

    if (!rawInput) {
        alert("Please enter at least one URL.");
        return;
    }

    // Extract the clean URLs
    const urls = extractUrls(rawInput);

    if (urls.length === 0) {
        alert("No valid URLs detected.");
        return;
    }

    // Display loading state
    resultsDiv.innerHTML = `<p style="color:#555;">Processing ${urls.length} links...</p>`;

    try {
        const response = await fetch("https://crowntalk-v2-0.onrender.com/comment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls })
        });

        const data = await response.json();

        resultsDiv.innerHTML = "";
        failedDiv.innerHTML = "";

        //-------------------------------------------
        // SHOW RESULTS
        //-------------------------------------------
        if (data.results && data.results.length > 0) {
            data.results.forEach((item) => {
                const block = document.createElement("div");
                block.className = "result-block";

                block.innerHTML = `
                    <p><a href="${item.url}" target="_blank">${item.url}</a></p>
                    <div class="comment-line">
                        <span>${item.comments[0]}</span>
                        <button onclick="copyText('${item.comments[0]}')">Copy</button>
                    </div>
                    <div class="comment-line">
                        <span>${item.comments[1]}</span>
                        <button onclick="copyText('${item.comments[1]}')">Copy</button>
                    </div>
                    <hr>
                `;
                resultsDiv.appendChild(block);
            });
        }

        //-------------------------------------------
        // SHOW FAILED URLS
        //-------------------------------------------
        if (data.failed && data.failed.length > 0) {
            let failHTML = "<h3>Failed</h3>";
            data.failed.forEach((url) => {
                failHTML += `<p><a href="${url}" target="_blank">${url}</a> — Unknown reason</p>`;
            });
            failedDiv.innerHTML = failHTML;
        }

    } catch (err) {
        resultsDiv.innerHTML = `<p style="color:red;">Server error. Try again.</p>`;
        console.error(err);
    }
});


//-------------------------------------------
// Copy button
//-------------------------------------------
function copyText(text) {
    navigator.clipboard.writeText(text);
    alert("Copied!");
}
