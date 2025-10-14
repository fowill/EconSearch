import os
from pathlib import Path

_default_index = os.getenv("PAPER_INDEX_PATH", "storage/paper_index.json")
PAPER_INDEX_PATH = Path(_default_index).expanduser()

OPENAI_MODEL = "gpt-4o-mini"
