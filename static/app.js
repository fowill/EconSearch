const ingestForm = document.getElementById("ingest-form");
const ingestStatus = document.getElementById("ingest-status");
const askForm = document.getElementById("ask-form");
const askStatus = document.getElementById("ask-status");
const answerContainer = document.getElementById("answer-container");
const answerEl = document.getElementById("answer");
const keywordsEl = document.getElementById("keywords");
const sourceListEl = document.getElementById("sources-list");

const askLogEl = document.getElementById("ask-log");

let askProgressTimeouts = [];
const askProgressSteps = [
    { delay: 0, text: "Generating search keywords..." },
    { delay: 1100, text: "Retrieving relevant papers..." },
    { delay: 2000, text: "Summarizing with the language model..." }
];

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

function clearAskProgressTimers() {
    if (askProgressTimeouts.length) {
        askProgressTimeouts.forEach((timer) => clearTimeout(timer));
        askProgressTimeouts = [];
    }
}

function resetAskLog() {
    if (!askLogEl) {
        return;
    }
    askLogEl.innerHTML = "";
    askLogEl.classList.remove("hidden");
}

function appendLog(message, type = "info") {
    if (!askLogEl) {
        return;
    }
    askLogEl.classList.remove("hidden");
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.textContent = message;
    askLogEl.appendChild(entry);
    askLogEl.scrollTop = askLogEl.scrollHeight;
}

function startAskProgress() {
    clearAskProgressTimers();
    resetAskLog();
    appendLog("Question submitted. Waiting for server acknowledgement...", "info");
    if (askStatus) {
        askStatus.textContent = "Processing question...";
        askStatus.className = "status info";
    }
    askProgressSteps.forEach((step) => {
        const timer = setTimeout(() => {
            appendLog(step.text, "info");
            if (askStatus) {
                askStatus.textContent = step.text;
                askStatus.className = "status info";
            }
        }, step.delay);
        askProgressTimeouts.push(timer);
    });
}

function stopAskProgress(message, statusClass) {
    clearAskProgressTimers();
    if (askStatus) {
        askStatus.textContent = message;
        askStatus.className = statusClass;
    }
}


ingestForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const pdfDir = document.getElementById("pdfDir").value.trim();
    const workersValue = document.getElementById("workers").value.trim();

    if (!pdfDir) {
        ingestStatus.textContent = "Using DEFAULT_PDF_DIR from server configuration...";
        ingestStatus.className = "status info";
    } else {
        ingestStatus.textContent = "Ingesting...";
        ingestStatus.className = "status";
    }

    try {
        const payload = { pdf_dir: pdfDir || null };
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
        clearAskProgressTimers();
        resetAskLog();
        appendLog("Please enter a question before asking.", "error");
        if (askStatus) {
            askStatus.textContent = "Please enter a question.";
            askStatus.className = "status error";
        }
        return;
    }

    startAskProgress();
    answerContainer.classList.add("hidden");
    if (sourceListEl) {
        sourceListEl.classList.add("hidden");
        sourceListEl.innerHTML = "";
    }

    try {
        const result = await postJSON("/ask", { question, top_k: topK });

        stopAskProgress("Answer ready.", "status success");

        const keywords = Array.isArray(result.keywords) ? result.keywords : [];
        const keywordPreview = keywords.slice(0, 3).join(", ") + (keywords.length > 3 ? ", ..." : "");
        appendLog(`Keywords ready (${keywords.length || 0}): ${keywordPreview || "None"}`, "success");

        answerEl.textContent = result.answer || "No answer returned.";

        keywordsEl.innerHTML = "";
        keywords.forEach((kw) => {
            const li = document.createElement("li");
            li.textContent = kw;
            keywordsEl.appendChild(li);
        });

        if (sourceListEl) {
            const sources = Array.isArray(result.sources) ? [...result.sources] : [];
            if (sources.length === 0) {
                appendLog("No matching papers found for this query.", "info");
                sourceListEl.innerHTML = '<p class="source-row-meta">No sources returned.</p>';
                sourceListEl.classList.remove("hidden");
            } else {
                appendLog(`Retrieved ${sources.length} candidate papers.`, "success");
                const sortedSources = sources.sort((a, b) => {
                    const yearA = typeof a.year === "number" ? a.year : Number.MAX_SAFE_INTEGER;
                    const yearB = typeof b.year === "number" ? b.year : Number.MAX_SAFE_INTEGER;
                    if (yearA === yearB) {
                        return (b.score || 0) - (a.score || 0);
                    }
                    return yearA - yearB;
                });

                sourceListEl.innerHTML = "";
                sortedSources.forEach((source) => {
                    const card = document.createElement("div");
                    card.className = "source-row";

                    const title = document.createElement("h4");
                    title.textContent = source.title || "Untitled";

                    const meta = document.createElement("div");
                    meta.className = "source-row-meta";

                    const journalSpan = document.createElement("span");
                    journalSpan.textContent = `Journal: ${source.journal || "Unknown"}`;

                    const yearSpan = document.createElement("span");
                    yearSpan.textContent = `Year: ${source.year ?? "N/A"}`;

                    const authorSpan = document.createElement("span");
                    const authorList = Array.isArray(source.authors) ? source.authors.filter(Boolean) : [];
                    const limitedAuthors = authorList.slice(0, 6);
                    const authorLabel = limitedAuthors.length
                        ? limitedAuthors.join(", ") + (authorList.length > 6 ? " et al." : "")
                        : "Unknown";
                    authorSpan.textContent = `Authors: ${authorLabel}`;

                    const scoreSpan = document.createElement("span");
                    const score = typeof source.score === "number" ? source.score.toFixed(3) : "N/A";
                    scoreSpan.textContent = `Similarity score: ${score}`;

                    meta.appendChild(journalSpan);
                    meta.appendChild(yearSpan);
                    meta.appendChild(authorSpan);
                    meta.appendChild(scoreSpan);

                    const abstract = document.createElement("p");
                    abstract.className = "source-row-abstract";
                    abstract.textContent = source.abstract || "No abstract available.";

                    const path = document.createElement("div");
                    path.className = "source-row-path";
                    path.innerHTML = `<strong>PDF:</strong> <code>${source.pdf_path || "Unknown path"}</code>`;

                    card.appendChild(title);
                    card.appendChild(meta);
                    card.appendChild(abstract);
                    card.appendChild(path);

                    if (source.pdf_path) {
                        const summaryLink = document.createElement("a");
                        summaryLink.href = `/summary?path=${encodeURIComponent(source.pdf_path)}`;
                        summaryLink.target = "_blank";
                        summaryLink.rel = "noopener";
                        summaryLink.className = "source-summary-link";
                        summaryLink.textContent = "Open LLM summary";
                        card.appendChild(summaryLink);
                    }

                    sourceListEl.appendChild(card);
                });

                sourceListEl.classList.remove("hidden");
            }
        }

        appendLog("Answer summarized and displayed.", "success");
        answerContainer.classList.remove("hidden");
    } catch (error) {
        stopAskProgress(`Failed to answer: ${error.message}`, "status error");
        appendLog(`Error: ${error.message}`, "error");
    }
});

