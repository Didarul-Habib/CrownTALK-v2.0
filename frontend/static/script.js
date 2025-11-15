const backendURL = "https://crowntalk-v2-0.onrender.com/comment";

document.getElementById("submitBtn").addEventListener("click", async () => {
    const output = document.getElementById("output");
    const input = document.getElementById("inputUrls").value.trim();

    if (!input) {
        output.innerHTML = "<p>No URLs provided.</p>";
        return;
    }

    const urls = input.split("\n").map(x => x.trim()).filter(Boolean);

    output.innerHTML = `
        <p><b>Processing ${urls.length} URL(s)...</b></p>
        <div id="live"></div>
        <div id="failed"></div>
    `;

    const liveBox = document.getElementById("live");
    const failBox = document.getElementById("failed");

    try {
        const res = await fetch(backendURL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls })
        });

        if (!res.ok) {
            output.innerHTML = `<p style="color:red;">Connection error. Try again.</p>`;
            return;
        }

        const data = await res.json();

        // STREAM SUCCESS
        data.results.forEach(r => {
            liveBox.innerHTML += `
                <div class="block">
                    <b>${r.url}</b><br><br>

                    • ${r.comments[0]}
                    <button class="copy-btn"
                        onclick="navigator.clipboard.writeText('${r.comments[0].replace(/'/g, "\\'")}')">
                        Copy
                    </button>
                    <br><br>

                    • ${r.comments[1]}
                    <button class="copy-btn"
                        onclick="navigator.clipboard.writeText('${r.comments[1].replace(/'/g, "\\'")}')">
                        Copy
                    </button>
                </div>
            `;
        });

        // STREAM FAILURES
        if (data.failed.length > 0) {
            failBox.innerHTML = `<h3 style="color:red;">Failed</h3>`;
            data.failed.forEach(f => {
                failBox.innerHTML += `❌ <b>${f.url}</b> — ${f.reason}<br>`;
            });
        }

    } catch (err) {
        output.innerHTML = `<p style="color:red;">Connection error. Try again.</p>`;
    }
});
