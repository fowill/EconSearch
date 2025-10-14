from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingest import ingest_folder
from llm import answer_with_context, generate_keywords
from search_engine import PaperSearchEngine, batch_load_fulltexts
from settings import PAPER_INDEX_PATH

load_dotenv()

app = FastAPI(title="EconSearch API", description="Keyword-driven paper QA over local PDFs.")

_engine: Optional[PaperSearchEngine] = None
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _get_engine(force_reload: bool = False) -> PaperSearchEngine:
    global _engine
    if force_reload or _engine is None:
        _engine = PaperSearchEngine(PAPER_INDEX_PATH)
    return _engine


class IngestRequest(BaseModel):
    pdf_dir: str = Field(..., description="Directory containing PDF files to ingest.")
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
    ingest_folder(request.pdf_dir, PAPER_INDEX_PATH, workers=request.workers)
    engine = _get_engine(force_reload=True)
    return IngestResponse(total_papers=len(engine.papers), index_path=str(PAPER_INDEX_PATH))


def _aggregate_search(question: str, top_k: int) -> Tuple[List[str], List[Dict[str, object]]]:
    keywords = generate_keywords(question, n_keywords=max(4, top_k * 2))
    engine = _get_engine()
    aggregated: Dict[str, Dict[str, object]] = {}
    for keyword in keywords:
        hits = engine.search(keyword, top_k=top_k * 3)
        for hit in hits:
            pdf_path = hit.get("pdf_path")
            if not pdf_path:
                continue
            entry = aggregated.setdefault(
                pdf_path,
                {
                    "metadata": {
                        "title": hit.get("title", ""),
                        "pdf_path": pdf_path,
                        "abstract": hit.get("abstract", ""),
                        "year": hit.get("year"),
                        "authors": hit.get("authors", []),
                        "keywords": hit.get("keywords", []),
                    },
                    "score": 0.0,
                },
            )
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
    full_texts = batch_load_fulltexts(papers, max_pages=12)

    contexts: List[str] = []
    for meta, full_text in zip(papers, full_texts):
        head = [
            f"Title: {meta.get('title')}",
            f"Year: {meta.get('year')}",
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
            score=float(sorted_items[idx]["score"]),
        )
        for idx, meta in enumerate(papers)
    ]

    return AskResponse(answer=answer, keywords=keywords, sources=sources)


@app.post("/reload")
def reload_index() -> Dict[str, str]:
    _get_engine(force_reload=True)
    return {"status": "reloaded"}
