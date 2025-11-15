//-------------------------------------------
// Extract clean URLs from user input
//-------------------------------------------
function extractUrls(inputText) {
    const lines = inputText.split("\n");
    const urls = [];

    for (let line of lines) {
        if (!line.trim()) continue;

        // Remove numbering (1., 2), 3 -, etc.)
        line = line.replace(/^\s*\d+[\.\)\-:]*\s*/, "").trim();

        // Extract actual URL
        const match = line.match(/https?:\/\/[^\s]+/);
        if (match) {
            let url = match[0].trim();
            url = url.replace(/[),.]+$/, ""); // remove trailing punctuation
            urls.push(url);
        }
    }

    return urls;
}


//-------------------------------------------
// Handle "Generate Comments"
//-------------------------------------------
document.getElementById("submitBtn").addEventListener("click", async function () {

    const rawInput = document.getElementById("inputUrls").value.trim();
    const outputDiv = document.getElementById("output");

    outputDiv.innerHTML = "";

    if (!rawInput) {
        alert("Please enter at least one URL.");
        return;
    }

    const urls = extractUrls(rawInput);

    if (urls.length === 0) {
        alert("No valid URLs found.");
        return;
    }

    outputDiv.innerHTML = `<p style="color:#777;">Processing ${urls.length} links...</p>`;

    try {
        const response = await fetch("https://crowntalk-v2-0.onrender.com/comment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls })
        });

        const data = await response.json();
        outputDiv.innerHTML = "";

        //-------------------------------------------
        // Show results
        //-------------------------------------------
        if (data.results && data.results.length > 0) {
            let html = "<h2>Results</h2>";

            data.results.forEach(item => {
                html += `
                    <div class="result-block">
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
                    </div>
                `;
            });

            outputDiv.innerHTML += html;
        }

        //-------------------------------------------
        // Show failed URLs
        //-------------------------------------------
        if (data.failed && data.failed.length > 0) {
            let fail = "<h2>Failed</h2>";

            data.failed.forEach(url => {
                fail += `<p><a href="${url}" target="_blank">${url}</a> — Unknown reason</p>`;
            });

            outputDiv.innerHTML += fail;
        }

    } catch (err) {
        outputDiv.innerHTML = `<p style="color:red;">Server error. Try again.</p>`;
        console.error(err);
    }
});


//-------------------------------------------
// Copy utility
//-------------------------------------------
function copyText(text) {
    navigator.clipboard.writeText(text);
    alert("Copied!");
}
