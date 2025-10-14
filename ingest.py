import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pypdf import PdfReader


@dataclass
class PaperMetadata:
    pdf_path: str
    title: str
    abstract: str
    year: Optional[int]
    authors: List[str]
    keywords: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "keywords": self.keywords,
        }


def _clean_parts(parts: Sequence[Optional[str]]) -> List[str]:
    cleaned: List[str] = []
    for item in parts:
        if not item:
            continue
        for chunk in re.split(r"[;,/]", str(item)):
            chunk = chunk.strip()
            if chunk and chunk.lower() != "none":
                cleaned.append(chunk)
    return cleaned


def _parse_year(meta: Dict[str, object]) -> Optional[int]:
    raw_candidates = [
        meta.get("/CreationDate"),
        meta.get("/ModDate"),
        meta.get("creationDate"),
        meta.get("modDate"),
        meta.get("created"),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        match = re.search(r"(19|20)\d{2}", str(raw))
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                continue
    return None


def _extract_preview(reader: PdfReader, max_pages: int = 4, max_chars: int = 5000) -> str:
    text_parts: List[str] = []
    for idx, page in enumerate(reader.pages):
        if idx >= max_pages:
            break
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            text_parts.append(txt)
        if sum(len(t) for t in text_parts) >= max_chars:
            break
    preview = "\n".join(text_parts).strip()
    return preview[:max_chars]


def _guess_title(meta: Dict[str, object], preview: str, pdf_path: Path) -> str:
    title_candidates = [
        meta.get("/Title"),
        getattr(meta, "title", None),
    ]
    for candidate in title_candidates:
        if candidate:
            return str(candidate).strip()
    lines = [ln.strip() for ln in preview.splitlines() if ln.strip()]
    if lines:
        return lines[0][:300]
    return pdf_path.stem


def _guess_authors(meta: Dict[str, object], preview: str) -> List[str]:
    from_meta = _clean_parts(
        [
            meta.get("/Author"),
            getattr(meta, "author", None),
        ]
    )
    if from_meta:
        return from_meta
    # try to read second line as author line if looks like comma-separated names
    lines = [ln.strip() for ln in preview.splitlines() if ln.strip()]
    if len(lines) > 1:
        possible = _clean_parts([lines[1]])
        if 1 <= len(possible) <= 6:  # heuristic guard
            return possible
    return []


def _guess_keywords(meta: Dict[str, object], preview: str) -> List[str]:
    meta_keywords = _clean_parts(
        [
            meta.get("/Keywords"),
            getattr(meta, "keywords", None),
            meta.get("/Subject"),
            getattr(meta, "subject", None),
        ]
    )
    if meta_keywords:
        return meta_keywords

    match = re.search(r"(?i)keywords?\s*[:\-]\s*(.+)", preview)
    if match:
        return _clean_parts([match.group(1)])
    return []


def _guess_abstract(preview: str) -> str:
    lowered = preview.lower()
    if "abstract" in lowered:
        match = re.search(r"(?is)abstract[:\s]*(.+?)(?:\n\s*\n|keywords?:|\Z)", preview)
        if match:
            abstract = match.group(1).strip()
            return abstract[:1500]
    # fallback: take the first 1500 chars of preview text
    return preview[:1500]


def _process_pdf(path_str: str) -> Optional[PaperMetadata]:
    pdf_path = Path(path_str)
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        print(f"failed to read {pdf_path}: {exc}")
        return None
    preview = _extract_preview(reader)
    if not preview:
        print(f"warning: no preview text extracted for {pdf_path}")
    raw_meta = reader.metadata or {}
    normalized_meta = {}
    # Normalize metadata keys for consistent lookup
    for key, value in dict(raw_meta).items():
        normalized_meta[key] = value
        normalized_meta[key.lstrip("/")] = value

    title = _guess_title(normalized_meta, preview, pdf_path)
    authors = _guess_authors(normalized_meta, preview)
    keywords = _guess_keywords(normalized_meta, preview)
    abstract = _guess_abstract(preview)
    year = _parse_year(normalized_meta)

    return PaperMetadata(
        pdf_path=str(pdf_path),
        title=title,
        abstract=abstract,
        year=year,
        authors=authors,
        keywords=keywords,
    )


def _load_existing(index_path: Path) -> Dict[str, Dict[str, object]]:
    if not index_path.exists():
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as fh:
            items = json.load(fh)
    except Exception:
        return {}
    return {item.get("pdf_path"): item for item in items if item.get("pdf_path")}


def ingest_folder(pdf_folder: str, out_index: str, workers: Optional[int] = None) -> None:
    pdf_dir = Path(pdf_folder)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {pdf_dir}")
    out_path = Path(out_index)
    existing = _load_existing(out_path)

    pdf_paths = sorted(str(p) for p in pdf_dir.rglob("*.pdf"))
    to_process = [p for p in pdf_paths if p not in existing]
    if not to_process:
        print("No new PDFs to process.")
        return

    max_workers = workers or min(4, os.cpu_count() or 1)
    results: List[PaperMetadata] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_pdf, path): path for path in to_process}
        for future in as_completed(futures):
            path = futures[future]
            try:
                paper = future.result()
                if paper:
                    results.append(paper)
                    print(f"Processed {path}")
                else:
                    print(f"Skipped {path}")
            except Exception as exc:
                print(f"Error processing {path}: {exc}")

    for paper in results:
        existing[paper.pdf_path] = paper.to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(existing.values(), key=lambda item: item["title"])
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(ordered)} papers to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs and store lightweight metadata.")
    parser.add_argument("--pdf-dir", required=True, help="Folder containing PDF files.")
    parser.add_argument(
        "--out",
        default="storage/paper_index.json",
        help="Path to the JSON index to create or update.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: min(4, cpu_count)).",
    )

    args = parser.parse_args()
    ingest_folder(args.pdf_dir, args.out, workers=args.workers)
