const input = document.getElementById("inputUrls");
const btn = document.getElementById("submitBtn");
const output = document.getElementById("output");
const progress = document.getElementById("progress");

btn.addEventListener("click", () => {
    const raw = input.value.trim();
    if (!raw) return;

    const urls = raw.split("\n").map(x => x.trim()).filter(Boolean);

    output.innerHTML = "";
    progress.innerHTML = "Starting…";

    const evtSource = new EventSource(
        "https://crowntalk-v2-0.onrender.com/stream?urls=" + encodeURIComponent(JSON.stringify(urls))
    );

    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // End of stream
        if (data.done) {
            progress.innerHTML = "✔ Completed!";
            evtSource.close();
            return;
        }

        // Update progress
        progress.innerHTML = `Batch ${data.batch}/${data.total} completed…`;

        // Add results
        appendBatch(data.results);
    };

    evtSource.onerror = () => {
        progress.innerHTML = "❌ Error contacting server.";
        evtSource.close();
    };
});

function appendBatch(results) {
    results.forEach(item => {
        const block = document.createElement("div");
        block.className = "tweet-block";

        block.innerHTML = `
            <p><strong>${item.url}</strong></p>
            ${renderComment(item.comments[0])}
            ${renderComment(item.comments[1])}
        `;

        output.appendChild(block);
    });
}

function renderComment(text) {
    const safe = text.replace(/'/g, "\\'");
    return `
        <div class="comment-line">
            <span>${text}</span>
            <button class="copyBtn" onclick="copyText('${safe}', this)">Copy</button>
        </div>
    `;
}

function copyText(text, btn) {
    navigator.clipboard.writeText(text).then(() => {
        btn.innerText = "Copied!";
        btn.style.background = "#28a745";
        setTimeout(() => {
            btn.innerText = "Copy";
            btn.style.background = "";
        }, 800);
    });
}
