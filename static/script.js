const backend = "https://crowntalk-v2-0.onrender.com/comment";

async function generateComments() {
    const input = document.getElementById("inputBox").value.trim();
    if (!input) return alert("Enter at least one URL");

    const urls = input.split("\n").map(u => u.trim()).filter(u => u !== "");

    document.getElementById("results").innerHTML = "";
    document.querySelector(".btn-generate").disabled = true;

    const response = await fetch(backend, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls })
    });

    const data = await response.json();
    console.log("Backend:", data);

    // SUCCESS RESULTS
    if (data.results?.length > 0) {
        data.results.forEach(item => {
            const div = `
                <div class="result-block">
                    <div class="result-url">
                        <a href="${item.url}" target="_blank">${item.url}</a>
                    </div>

                    <div class="comment-line">${item.comments[0]}</div>
                    <div class="comment-line">${item.comments[1]}</div>
                </div>
            `;
            document.getElementById("results").innerHTML += div;
        });
    }

    // FAILED LINKS
    if (data.failed?.length > 0) {
        document.getElementById("results").innerHTML += `
            <div class="result-block">
                <strong>Failed:</strong><br>
                ${data.failed.join("<br>")}
            </div>
        `;
    }

    document.querySelector(".btn-generate").disabled = false;
}

// THEME SWITCH
function toggleTheme() {
    document.body.classList.toggle("theme-sunset");
}
