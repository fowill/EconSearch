from pathlib import Path
import re
from html import escape
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingest import ingest_folder
from llm import answer_with_context, generate_keywords, summarize_document
from search_engine import PaperSearchEngine, batch_load_fulltexts
from settings import PAPER_INDEX_PATH, DEFAULT_PDF_DIR

load_dotenv()

app = FastAPI(title="EconSearch API", description="Keyword-driven paper QA over local PDFs.")

_engine: Optional[PaperSearchEngine] = None
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SMALL_WORDS = {
    "of",
    "and",
    "the",
    "in",
    "on",
    "for",
    "to",
    "a",
    "an",
    "at",
    "by",
    "with",
    "or",
    "from",
    "per",
}


def _get_engine(force_reload: bool = False) -> PaperSearchEngine:
    global _engine
    if force_reload or _engine is None:
        _engine = PaperSearchEngine(PAPER_INDEX_PATH)
    return _engine


def _format_title_case(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    stripped = text.strip()
    if not stripped:
        return stripped
    tokens = re.split(r"(\s+)", stripped.lower())
    result: List[str] = []
    first_word = True
    for token in tokens:
        if not token:
            continue
        if token.isspace():
            result.append(token)
            continue
        segments = token.split("-")
        rebuilt = []
        for idx, segment in enumerate(segments):
            if not segment:
                continue
            lower = segment.lower()
            capitalize = first_word or lower not in SMALL_WORDS
            if capitalize:
                rebuilt.append(lower[:1].upper() + lower[1:])
            else:
                rebuilt.append(lower)
            first_word = False
        result.append("-".join(rebuilt))
        first_word = False
    return "".join(result)


def _find_metadata_for_path(pdf_path: Path) -> Optional[Dict[str, object]]:
    engine = _get_engine()
    target = str(pdf_path)
    for paper in engine.papers:
        candidate = paper.get("pdf_path")
        if not candidate:
            continue
        try:
            candidate_path = str(Path(candidate).resolve())
        except Exception:
            candidate_path = str(Path(candidate))
        if candidate_path == target:
            return paper
    return None


class IngestRequest(BaseModel):
    pdf_dir: Optional[str] = Field(None, description="Directory containing PDF files to ingest.")
    workers: Optional[int] = Field(
        None, description="Number of worker processes to use for ingestion."
    )


class IngestResponse(BaseModel):
    total_papers: int
    index_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(
        3, ge=1, le=10, description="Maximum number of papers to feed into the LLM."
    )


class Source(BaseModel):
    title: str
    pdf_path: str
    score: float
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    journal: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    keywords: List[str]
    sources: List[Source]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=503, detail="UI assets not found. Regenerate static files.")
    return FileResponse(index_file)


@app.get("/info")
def info() -> Dict[str, str]:
    return {
        "message": "EconSearch API is running. Use /health, /ingest, /ask, or /reload for programmatic access."
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    pdf_dir = request.pdf_dir or (str(DEFAULT_PDF_DIR) if DEFAULT_PDF_DIR else None)
    if not pdf_dir:
        raise HTTPException(status_code=400, detail="PDF directory not provided. Supply a path or set DEFAULT_PDF_DIR.")
    try:
        pdf_path_obj = Path(pdf_dir).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF directory path.")
    if not pdf_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"PDF directory not found: {pdf_dir}")
    ingest_folder(str(pdf_path_obj), PAPER_INDEX_PATH, workers=request.workers)
    engine = _get_engine(force_reload=True)
    return IngestResponse(total_papers=len(engine.papers), index_path=str(PAPER_INDEX_PATH))


def _aggregate_search(question: str, top_k: int) -> Tuple[List[str], List[Dict[str, object]]]:
    requested = min(6, max(3, top_k + 2))
    raw_keywords = generate_keywords(question, n_keywords=requested)
    keywords: List[str] = []
    seen: set[str] = set()
    for kw in raw_keywords:
        cleaned = kw.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(cleaned)
    if not keywords:
        keywords = [question]

    engine = _get_engine()
    aggregated: Dict[str, Dict[str, object]] = {}
    per_query_k = max(top_k, 4)
    for keyword in keywords:
        hits = engine.search(keyword, top_k=per_query_k)
        for hit in hits:
            pdf_path = hit.get("pdf_path")
            if not pdf_path:
                continue
            journal = hit.get("journal")
            if isinstance(journal, str):
                journal = _format_title_case(journal)
            metadata = {
                "title": hit.get("title", ""),
                "pdf_path": pdf_path,
                "abstract": hit.get("abstract", ""),
                "year": hit.get("year"),
                "authors": hit.get("authors", []),
                "keywords": hit.get("keywords", []),
                "journal": journal,
            }
            entry = aggregated.setdefault(
                pdf_path,
                {
                    "metadata": metadata,
                    "score": 0.0,
                },
            )
            if entry["metadata"].get("journal") in (None, "", "Unknown") and journal:
                entry["metadata"]["journal"] = journal
            entry["score"] += float(hit.get("score", 0.0))
    sorted_hits = [
        item[1] for item in sorted(aggregated.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]
    ]
    return keywords, sorted_hits


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        keywords, sorted_items = _aggregate_search(request.question, request.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No paper index found. Run /ingest first.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not sorted_items:
        return AskResponse(answer="No relevant documents found.", keywords=keywords, sources=[])
    papers = [item["metadata"] for item in sorted_items]
    full_texts = batch_load_fulltexts(papers, max_pages=4, max_chars=8000)

    contexts: List[str] = []
    for meta, full_text in zip(papers, full_texts):
        head = [
            f"Title: {meta.get('title')}",
            f"Year: {meta.get('year')}",
            f"Journal: {meta.get('journal')}" if meta.get("journal") else "Journal: Unknown",
            f"Authors: {', '.join(meta.get('authors', []))}" if meta.get("authors") else "Authors: Unknown",
            f"Keywords: {', '.join(meta.get('keywords', []))}" if meta.get("keywords") else "Keywords: None",
            "",
            meta.get("abstract") or "",
            "",
            full_text[:4000],
        ]
        contexts.append("\n".join(part for part in head if part is not None))

    answer = answer_with_context(request.question, contexts)

    sources = [
        Source(
            title=meta.get("title", ""),
            pdf_path=meta.get("pdf_path", ""),
            abstract=meta.get("abstract"),
            year=meta.get("year"),
            authors=meta.get("authors", []),
            keywords=meta.get("keywords", []),
            journal=meta.get("journal"),
            score=float(sorted_items[idx]["score"]),
        )
        for idx, meta in enumerate(papers)
    ]

    return AskResponse(answer=answer, keywords=keywords, sources=sources)



@app.get("/summary", response_class=HTMLResponse)
def render_summary(path: str) -> HTMLResponse:
    if not path:
        raise HTTPException(status_code=400, detail="Missing path parameter.")
    try:
        pdf_path = Path(path).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path parameter.")

    metadata = _find_metadata_for_path(pdf_path)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Paper not found in current index.")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found on disk.")

    try:
        text = PaperSearchEngine.load_fulltext(str(pdf_path), max_pages=None, max_chars=40000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {exc}")

    title = metadata.get("title") or pdf_path.stem
    journal = metadata.get("journal") or "Unknown"
    journal_fmt = _format_title_case(journal) if journal else "Unknown"
    year = metadata.get("year") or "N/A"
    authors = metadata.get("authors") or []
    authors_display = ", ".join(authors) if authors else "Unknown"

    summary_text = summarize_document(title, text)
    summary_html = escape(summary_text).replace("\n", "<br>")

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <title>Summary | {escape(title)}</title>
    <link rel=\"stylesheet\" href=\"/static/styles.css\">
    <style>
        body {{ background: #f4f6fb; margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        .summary-container {{ max-width: 900px; margin: 2rem auto; background: #ffffff; border-radius: 12px; padding: 2rem 2.5rem; box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12); }}
        .summary-actions {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }}
        .summary-actions a {{ color: #2563eb; font-weight: 600; text-decoration: none; }}
        .summary-actions a:hover {{ text-decoration: underline; }}
        .summary-meta {{ color: #475569; font-size: 0.95rem; display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.8rem; }}
        .summary-text {{ line-height: 1.65; color: #1f2937; background: #f8fafc; border-radius: 12px; padding: 1.3rem; border: 1px solid #e2e8f0; }}
        .summary-text code {{ background: rgba(15, 23, 42, 0.06); padding: 0.1rem 0.3rem; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class=\"summary-container\">
        <div class=\"summary-actions\">
            <a href=\"/\">&#8592; Back to search</a>
            <a href=\"#\" onclick=\"window.close(); return false;\">Close tab</a>
        </div>
        <h1>{escape(title)}</h1>
        <p class=\"summary-meta\">
            <span><strong>Journal:</strong> {escape(journal_fmt)}</span>
            <span><strong>Year:</strong> {escape(str(year))}</span>
            <span><strong>Authors:</strong> {escape(authors_display)}</span>
        </p>
        <section>
            <h2>LLM Summary</h2>
            <div class=\"summary-text\">{summary_html}</div>
        </section>
        <section style=\"margin-top: 1.5rem;\">
            <h2>Document</h2>
            <p><strong>PDF path:</strong> <code>{escape(str(pdf_path))}</code></p>
        </section>
    </div>
</body>
</html>"""

    return HTMLResponse(content=html_content)

@app.post("/reload")
def reload_index() -> Dict[str, str]:
    _get_engine(force_reload=True)
    return {"status": "reloaded"}
