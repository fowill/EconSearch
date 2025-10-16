import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pypdf import PdfReader


@dataclass
class PaperMetadata:
    pdf_path: str
    title: str
    abstract: str
    year: Optional[int]
    authors: List[str]
    keywords: List[str]
    journal: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": self.pdf_path,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "keywords": self.keywords,
            "journal": self.journal,
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


HEADER_STOPWORDS = {
    "journal",
    "review",
    "volume",
    "vol",
    "issue",
    "no",
    "number",
    "issn",
    "doi",
    "copyright",
    "press",
    "association",
    "conference",
}

MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

AFFILIATION_KEYWORDS = {
    "university",
    "college",
    "school",
    "department",
    "institute",
    "laboratory",
    "lab",
    "centre",
    "center",
    "academy",
    "research",
    "programme",
    "program",
}

FOOTNOTE_PREFIXES = ("*", "\u2020", "\u2021", "\u00A7")

JOURNAL_KEYWORDS = {
    "journal",
    "review",
    "quarterly",
    "economics",
    "finance",
    "studies",
    "letters",
    "magazine",
    "bulletin",
    "papers",
    "econometrica",
    "economic",
}

ACRONYM_WHITELIST = {
    "gdp",
    "us",
    "usa",
    "uk",
    "eu",
    "oecd",
    "un",
    "esg",
    "iv",
    "irl",
    "cpi",
    "nber",
    "ceo",
    "roe",
    "roa",
    "epa",
    "sec",
}

ROMAN_NUMERALS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}

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


def _normalize_token(token: str) -> str:
    base = token.strip()
    if not base:
        return ""
    base = base.replace("\u2019", "'").replace("`", "'")
    if "-" in base:
        parts = [_normalize_token(part) for part in base.split("-")]
        parts = [part for part in parts if part]
        return "-".join(parts)

    core = re.sub(r"[^A-Za-z']+", "", base)
    if not core:
        return ""
    lower = core.lower()
    if lower in ACRONYM_WHITELIST or lower in ROMAN_NUMERALS:
        return core.upper()
    return core[0].upper() + core[1:].lower() if len(core) > 1 else core.upper()


def _normalize_author_name(name: str) -> Optional[str]:
    name = name.strip(" ,;")
    if not name:
        return None
    name = re.sub(r"[\*\u2020\u2021\u00A7]+", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    name = name.replace("\u2013", "-").replace("\u2014", "-")
    tokens = [tok for tok in name.split() if tok]
    normalized_parts: List[str] = []
    for token in tokens:
        normalized = _normalize_token(token)
        if normalized:
            normalized_parts.append(normalized)
    normalized_name = " ".join(normalized_parts).strip(" ,;")
    return normalized_name or None


def _smart_title_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    tokens = re.split(r"(\s+)", text.lower())
    result: List[str] = []
    first_word = True
    for token in tokens:
        if not token:
            continue
        if token.isspace():
            result.append(token)
            continue
        parts = token.split("-")
        new_parts = []
        for idx, part in enumerate(parts):
            if not part:
                continue
            lower = part.lower()
            capitalize = first_word or lower not in SMALL_WORDS
            if capitalize:
                new_part = lower[:1].upper() + lower[1:]
            else:
                new_part = lower
            new_parts.append(new_part)
            first_word = False
        result.append("-".join(new_parts))
        first_word = False
    return "".join(result)


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


def _uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def _is_noise_line(line: str) -> bool:
    if not line:
        return False
    lowered = line.lower()
    if line.startswith(FOOTNOTE_PREFIXES):
        return True
    if lowered.startswith("doi") or lowered.startswith("http"):
        return True
    if len(line) <= 3 and line.isdigit():
        return True
    if re.fullmatch(r"[\d\-\s]+", line):
        return True
    tokens = [token.strip(".,;:") for token in lowered.split()]
    if not tokens:
        return False
    if any(token in AFFILIATION_KEYWORDS for token in tokens):
        return True
    has_header_keyword = any(token in HEADER_STOPWORDS for token in tokens)
    has_month = any(token in MONTH_NAMES for token in tokens)
    has_digits = any(any(ch.isdigit() for ch in token) for token in tokens)
    uppercase_ratio = _uppercase_ratio(line)
    if uppercase_ratio > 0.8 and (has_header_keyword or has_month or has_digits):
        return True
    if has_header_keyword and has_month:
        return True
    return False


def _prepare_preview_lines(preview: str) -> List[str]:
    processed: List[str] = []
    for raw in preview.splitlines():
        line = raw.strip()
        if not line:
            processed.append("")
            continue
        if _is_noise_line(line):
            continue
        processed.append(line)
    return processed


def _split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_author_names(text: str) -> List[str]:
    if not text:
        return []
    cleaned = re.sub(r"[\*\u2020\u2021\u00A7]+", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("&", ",")
    parts = [part.strip(" ,;") for part in cleaned.split(",") if part.strip(" ,;")]
    authors: List[str] = []
    for part in parts:
        lower = part.lower()
        if any(keyword in lower for keyword in AFFILIATION_KEYWORDS):
            continue
        if any(ch.isdigit() for ch in part):
            continue
        if len(part.split()) > 7:
            continue
        normalized = _normalize_author_name(part)
        if normalized and normalized not in authors:
            authors.append(normalized)
    return authors


def _guess_journal(preview: str, meta: Dict[str, object]) -> Optional[str]:
    candidates = [
        meta.get("Journal"),
        meta.get("/Journal"),
        meta.get("journal"),
        meta.get("/Subject"),
        meta.get("Subject"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).strip()
        if text:
            return text

    lines = [line.strip() for line in preview.splitlines() if line.strip()]
    for line in lines[:12]:
        normalized = re.sub(r"\s+", " ", line).strip("\u2022- ")
        lowered = normalized.lower()
        if any(keyword in lowered for keyword in JOURNAL_KEYWORDS):
            if "\u2022" in normalized:
                normalized = normalized.split("\u2022")[0].strip()
            return normalized
    return None


def _find_abstract_index(lines: List[str]) -> Optional[int]:
    for idx, line in enumerate(lines):
        normalized = line.lower().strip(" :")
        if normalized == "abstract" or normalized.startswith("abstract "):
            return idx
    return None


def _select_title_and_authors(lines: List[str]) -> Tuple[Optional[str], List[str]]:
    abstract_idx = _find_abstract_index(lines)
    candidate_lines = lines[:abstract_idx] if abstract_idx is not None else lines[:40]
    blocks = _split_blocks(candidate_lines)

    title: Optional[str] = None
    authors: List[str] = []
    title_block_idx: Optional[int] = None
    author_lines: List[str] = []

    for idx, block in enumerate(blocks):
        joined = " ".join(block)
        tokens = [token.strip(".,;:").lower() for token in joined.split()]
        if len(tokens) < 3:
            continue
        if any(token in HEADER_STOPWORDS for token in tokens):
            continue
        if tokens and tokens[0].isdigit():
            continue
        title = block[0]
        if len(block) > 1:
            author_lines.extend(block[1:])
        title_block_idx = idx
        break

    if title_block_idx is not None:
        for block in blocks[title_block_idx + 1 :]:
            joined = " ".join(block)
            if not joined:
                continue
            lowered = joined.lower()
            if "abstract" in lowered:
                break
            author_lines.extend(block)
            parsed = _parse_author_names(" ".join(author_lines))
            if parsed:
                authors = parsed
                break

    if not authors and author_lines:
        authors = _parse_author_names(" ".join(author_lines))

    return title, authors


SECTION_STOPWORDS = {
    "introduction",
    "background",
    "methods",
    "data",
    "results",
    "conclusion",
    "conclusions",
    "discussion",
    "literature review",
    "related literature",
    "model",
    "theory",
}


def _is_section_heading(line: str) -> bool:
    normalized = line.lower().strip(" :")
    if normalized in SECTION_STOPWORDS:
        return True
    uppercase_ratio = _uppercase_ratio(line)
    if uppercase_ratio > 0.85 and len(line.split()) <= 12:
        return True
    return False


def _extract_abstract_from_lines(lines: List[str], abstract_idx: Optional[int]) -> Optional[str]:
    if abstract_idx is None:
        return None
    abstract_lines: List[str] = []
    for line in lines[abstract_idx + 1 :]:
        if not line:
            if abstract_lines:
                break
            continue
        if _is_section_heading(line):
            break
        abstract_lines.append(line)
        if len(" ".join(abstract_lines)) >= 1500:
            break
    if not abstract_lines:
        return None
    return " ".join(abstract_lines)[:1500]


def _normalize_abstract_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""

    words = cleaned.split(" ")
    normalized_words: List[str] = []
    for word in words:
        stripped = re.sub(r"[^A-Za-z]", "", word)
        if len(stripped) >= 4 and stripped.isupper():
            lowered = word.lower()
            word = lowered[0].upper() + lowered[1:] if lowered else word
        normalized_words.append(word)

    normalized = " ".join(normalized_words)
    if not normalized:
        return normalized

    normalized = normalized[0].upper() + normalized[1:]

    def capitalise(match: re.Match) -> str:
        prefix = match.group(1)
        char = match.group(2)
        return f"{prefix}{char.upper()}"

    normalized = re.sub(r"([.!?]\s+)([a-z])", capitalise, normalized)
    return normalized


def _extract_metadata_from_preview(
    preview: str, meta: Dict[str, object], pdf_path: Path
) -> Tuple[str, List[str], str, Optional[str]]:
    journal = _guess_journal(preview, meta)
    lines = _prepare_preview_lines(preview)
    title, authors = _select_title_and_authors(lines)
    abstract_idx = _find_abstract_index(lines)
    abstract = _extract_abstract_from_lines(lines, abstract_idx)

    title_candidates = [
        title,
        meta.get("/Title"),
        getattr(meta, "title", None),
        meta.get("Title"),
    ]
    resolved_title = None
    for candidate in title_candidates:
        if candidate:
            resolved_title = str(candidate).strip()
            if resolved_title:
                break
    if not resolved_title:
        resolved_title = pdf_path.stem

    if not authors:
        authors = _clean_parts(
            [
                meta.get("/Author"),
                getattr(meta, "author", None),
                meta.get("Author"),
            ]
        )

    if not abstract:
        abstract = preview[:1500]
    abstract = _normalize_abstract_text(abstract)

    if journal:
        journal = _smart_title_case(journal)

    return resolved_title, authors, abstract, journal


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

    title, authors, abstract, journal = _extract_metadata_from_preview(preview, normalized_meta, pdf_path)
    keywords = _guess_keywords(normalized_meta, preview)
    year = _parse_year(normalized_meta)

    return PaperMetadata(
        pdf_path=str(pdf_path),
        title=title,
        abstract=abstract,
        year=year,
        authors=authors,
        keywords=keywords,
        journal=journal,
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
