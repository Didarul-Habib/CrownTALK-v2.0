const API_URL = "https://crowntalk-v2-0.onrender.com/comment";

function generateComments() {
    let raw = document.getElementById("input").value.trim();
    if (!raw) return alert("Please paste tweet links.");

    let urls = raw.split("\n").map(x => x.trim()).filter(x => x.length > 3);

    document.getElementById("generate").classList.add("hidden");
    document.getElementById("stop").classList.remove("hidden");

    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("generate").classList.remove("hidden");
        document.getElementById("stop").classList.add("hidden");

        let output = document.getElementById("output");
        output.innerHTML = "";

        data.results.forEach(item => {
            let block = document.createElement("div");
            block.className = "result-block";

            block.innerHTML = `
                <div class="result-url">
                    <a href="${item.url}" target="_blank">${item.url}</a>
                </div>
                <div class="comment-line">${item.comments[0]}</div>
                <div class="comment-line">${item.comments[1]}</div>
            `;

            output.appendChild(block);
        });

        if (data.failed.length > 0) {
            let failText = document.createElement("p");
            failText.style.color = "red";
            failText.innerHTML = "Failed URLs:<br>" + data.failed.join("<br>");
            output.appendChild(failText);
        }
    })
    .catch(err => {
        alert("Server error. Try again.");
        document.getElementById("generate").classList.remove("hidden");
        document.getElementById("stop").classList.add("hidden");
    });
}

function toggleTheme() {
    document.body.classList.toggle("theme-sunset");
}
