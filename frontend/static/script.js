const backendURL = "https://crowntalk-v2-0.onrender.com/comment";

document.getElementById("generateBtn").addEventListener("click", () => {
    const raw = document.getElementById("urlInput").value.trim();
    if (!raw) return;

    const urls = raw.split(/\n+/).map(u => u.trim()).filter(Boolean);

    document.getElementById("progress").innerHTML = "Processing...";
    document.getElementById("results").innerHTML = "";
    document.getElementById("failed").innerHTML = "";

    fetch(backendURL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ urls })
    })
    .then(res => res.json())
    .then(data => {
        const batches = data.batches;

        batches.forEach((batch, idx) => {
            document.getElementById("progress").innerHTML =
                `Batch ${batch.batch} done (${idx+1}/${batches.length})`;

            batch.results.forEach(item => {
                const block = document.createElement("div");
                block.className = "result-block";

                block.innerHTML = `
                    <a href="${item.url}" target="_blank">${item.url}</a><br><br>
                    <div>
                        ${item.comments[0]}
                        <button class="copy-btn" onclick="copyText('${item.comments[0]}')">copy</button>
                    </div>
                    <div>
                        ${item.comments[1]}
                        <button class="copy-btn" onclick="copyText('${item.comments[1]}')">copy</button>
                    </div>
                `;
                document.getElementById("results").appendChild(block);
            });

            batch.failed.forEach(item => {
                const f = document.createElement("div");
                f.className = "result-block";
                f.style.borderLeft = "4px solid red";
                f.innerHTML = `Failed: ${item.url} — ${item.reason}`;
                document.getElementById("failed").appendChild(f);
            });
        });

        document.getElementById("progress").innerHTML = "All batches completed!";
    });
});

function copyText(text) {
    navigator.clipboard.writeText(text);
}
