const ingestForm = document.getElementById("ingest-form");
const ingestStatus = document.getElementById("ingest-status");
const askForm = document.getElementById("ask-form");
const askStatus = document.getElementById("ask-status");
const answerContainer = document.getElementById("answer-container");
const answerEl = document.getElementById("answer");
const keywordsEl = document.getElementById("keywords");
const sourcesEl = document.getElementById("sources");

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || response.statusText);
    }
    return response.json();
}

ingestForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const pdfDir = document.getElementById("pdfDir").value.trim();
    const workersValue = document.getElementById("workers").value.trim();

    if (!pdfDir) {
        ingestStatus.textContent = "Please provide a PDF directory path.";
        ingestStatus.className = "status error";
        return;
    }

    ingestStatus.textContent = "Ingesting...";
    ingestStatus.className = "status";

    try {
        const payload = { pdf_dir: pdfDir };
        if (workersValue) {
            payload.workers = Number(workersValue);
        }
        const result = await postJSON("/ingest", payload);
        ingestStatus.textContent = `Ingested successfully. Total papers indexed: ${result.total_papers}`;
        ingestStatus.className = "status success";
    } catch (error) {
        ingestStatus.textContent = `Failed to ingest: ${error.message}`;
        ingestStatus.className = "status error";
    }
});

askForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = document.getElementById("question").value.trim();
    const topK = Number(document.getElementById("topK").value) || 3;

    if (!question) {
        askStatus.textContent = "Please enter a question.";
        askStatus.className = "status error";
        return;
    }

    askStatus.textContent = "Thinking...";
    askStatus.className = "status";
    answerContainer.classList.add("hidden");

    try {
        const result = await postJSON("/ask", { question, top_k: topK });

        askStatus.textContent = "Got an answer.";
        askStatus.className = "status success";

        answerEl.textContent = result.answer || "No answer returned.";

        keywordsEl.innerHTML = "";
        (result.keywords || []).forEach((kw) => {
            const li = document.createElement("li");
            li.textContent = kw;
            keywordsEl.appendChild(li);
        });

        sourcesEl.innerHTML = "";
        (result.sources || []).forEach((source) => {
            const card = document.createElement("div");
            card.className = "source-card";

            const title = document.createElement("h4");
            title.textContent = source.title || "Untitled";

            const meta = document.createElement("p");
            const parts = [];
            if (source.year) parts.push(`Year: ${source.year}`);
            if (source.score !== undefined) parts.push(`Score: ${source.score.toFixed(3)}`);
            if (source.authors?.length) parts.push(`Authors: ${source.authors.join(", ")}`);
            meta.textContent = parts.join(" · ");

            const abstract = document.createElement("p");
            abstract.textContent = source.abstract || "No abstract available.";

            const path = document.createElement("p");
            path.innerHTML = `<strong>PDF:</strong> <code>${source.pdf_path}</code>`;

            card.appendChild(title);
            card.appendChild(meta);
            card.appendChild(abstract);
            card.appendChild(path);

            sourcesEl.appendChild(card);
        });

        answerContainer.classList.remove("hidden");
    } catch (error) {
        askStatus.textContent = `Failed to answer: ${error.message}`;
        askStatus.className = "status error";
    }
});
