import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PaperSearchEngine:
    """Search over lightweight paper metadata and fetch full text on demand."""

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Paper index not found at {self.index_path}. Run ingest first."
            )
        with open(self.index_path, "r", encoding="utf-8") as fh:
            self.papers: List[Dict[str, object]] = json.load(fh)
        self._build_vector_store()

    def _build_vector_store(self) -> None:
        corpus = [self._compose_search_text(p) for p in self.papers]
        if not corpus:
            raise ValueError("Paper index is empty. Ingest PDFs before searching.")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=4096,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def _compose_search_text(paper: Dict[str, object]) -> str:
        parts: List[str] = []
        for key in ("title", "abstract", "journal"):
            value = paper.get(key)
            if value:
                parts.append(str(value))
        for key in ("authors", "keywords"):
            values = paper.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                parts.extend(str(v) for v in values)
        return " ".join(parts)

    def reload(self) -> None:
        """Reload the index file from disk and rebuild the vector store."""
        self._load_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        """Return top_k papers with cosine similarity scores."""
        if not query:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if top_k <= 0:
            top_k = len(self.papers)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results: List[Dict[str, object]] = []
        for idx in top_indices:
            paper = dict(self.papers[idx])
            paper["score"] = float(sims[idx])
            results.append(paper)
        return results

    @staticmethod
    @lru_cache(maxsize=128)
    def load_fulltext(
        pdf_path: str, max_pages: Optional[int] = None, max_chars: Optional[int] = None
    ) -> str:
        """Extract text from the referenced PDF. Cached to avoid repeated reads."""
        reader = PdfReader(pdf_path)
        pages = list(reader.pages)
        text_segments: List[str] = []
        total_chars = 0
        for page in pages[: max_pages or len(pages)]:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_segments.append(text)
                total_chars += len(text)
                if max_chars and total_chars >= max_chars:
                    break
        combined = "\n".join(text_segments)
        if max_chars:
            return combined[:max_chars]
        return combined


def batch_load_fulltexts(
    papers: Iterable[Dict[str, object]],
    max_pages: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[str]:
    """Helper to load full texts for a list of paper metadata entries."""
    texts: List[str] = []
    for paper in papers:
        pdf_path = paper.get("pdf_path")
        if not pdf_path:
            texts.append("")
            continue
        try:
            text = PaperSearchEngine.load_fulltext(
                str(pdf_path), max_pages=max_pages, max_chars=max_chars
            )
        except Exception:
            text = ""
        texts.append(text)
    return texts


__all__ = ["PaperSearchEngine", "batch_load_fulltexts"]
