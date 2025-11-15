document.getElementById("submitBtn").addEventListener("click", async () => {
    const output = document.getElementById("output");
    const rawText = document.getElementById("inputUrls").value.trim();

    if (!rawText) {
        output.innerHTML = "<p>Please enter URLs.</p>";
        return;
    }

    const urls = rawText.split("\n").map(u => u.trim()).filter(Boolean);

    output.innerHTML = "<p>Processing... Please wait.</p>";

    try {
        const res = await fetch("https://crowntalk-v2-0.onrender.com/comment", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ urls })
        });

        const data = await res.json();

        if (!res.ok) {
            output.innerHTML = `<p>Error: ${data.error}</p>`;
            return;
        }

        let html = "";

        data.results.forEach(item => {
            html += `
                <div class="card">
                    <h3>${item.url}</h3>
                    <p><strong>Tweet:</strong> ${item.tweet}</p>
                    <p><strong>Comment:</strong> ${item.comment}</p>
                </div>
            `;
        });

        if (data.failed.length > 0) {
            html += "<h2>Failed</h2>";
            data.failed.forEach(f => {
                html += `<p>${f.url} — ${f.reason}</p>`;
            });
        }

        output.innerHTML = html;

    } catch (err) {
        output.innerHTML = "<p>Server error. Try again.</p>";
    }
});
