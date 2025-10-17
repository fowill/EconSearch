import os
from pathlib import Path

_default_index = os.getenv("PAPER_INDEX_PATH", "storage/paper_index.json")
PAPER_INDEX_PATH = Path(_default_index).expanduser()

OPENAI_MODEL = "gpt-4o-mini"

_default_pdf_dir = os.getenv("DEFAULT_PDF_DIR", "")
DEFAULT_PDF_DIR = Path(_default_pdf_dir).expanduser() if _default_pdf_dir else None
