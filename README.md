EconSearch
==========

Minimal PDF ingestion + question answering pipeline focused on the essentials:

1. Ingest PDFs into a compact local index containing title, year, authors, keywords, and abstract.
2. Ask questions through a FastAPI service where the LLM:
   - generates English keywords for the query,
   - retrieves matching papers from the local index,
   - streams full PDF text on demand, and
   - answers using those documents.

Setup
-----
Assuming the `econsearch` micromamba environment:

```powershell
micromamba activate econsearch
pip install -r requirements.txt
```

Configuration
-------------
Here is an example `.env` configuration that covers both supported providers:

```ini
# Choose the provider to talk to: shubiaobiao | deepseek
LLM_PROVIDER=shubiaobiao

# Shared secret key (required for either provider)
OPENAI_API_KEY=sk-your-real-key

# Provider defaults (change only if you need a different endpoint/model)
SHUBIAOBIAO_BASE_URL=https://api.shubiaobiao.cn/v1/
SHUBIAOBIAO_MODEL=gpt-4o-mini
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/
DEEPSEEK_MODEL=deepseek-chat

# Optional explicit overrides (normally leave blank)
OPENAI_BASE_URL=
OPENAI_MODEL=
```

Set `LLM_PROVIDER` to `deepseek` if you want to switch. Restart the app after editing `.env` so the new values are picked up.

Copy `.env.example` to `.env` (or export the variables manually) and fill in your private values:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.shubiaobiao.cn/v1/
OPENAI_MODEL=gpt-4o-mini
```

All secrets are read from the environment at runtime, so nothing sensitive needs to live in the codebase.

Ingestion
---------
Build or update the metadata index (only metadata is stored, no full text persisted):

```powershell
micromamba run -n econsearch python ingest.py --pdf-dir "D:\path\to\pdfs" --out storage/paper_index.json
```

API & UI
--------
Serve the FastAPI app (Uvicorn example):

```powershell
micromamba run -n econsearch uvicorn app:app --host 0.0.0.0 --port 8000
```

Opening `http://localhost:8000/` loads a lightweight web UI that lets you trigger ingestion and ask questions without crafting manual HTTP calls.

Available endpoints:

- `GET /` : single-page UI for ingestion and Q&A.
- `GET /health` : readiness probe.
- `GET /info` : JSON endpoint summarizing available routes.
- `POST /ingest` : trigger ingestion from a directory.
- `POST /ask` : get an LLM answer plus retrieved keywords and sources.
- `POST /reload` : rebuild the in-memory search index from stored metadata.

Implementation Notes
--------------------
- `ingest.py` extracts metadata via `pypdf` and writes a lightweight JSON index.
- `search_engine.py` performs TF-IDF search over the metadata and loads full text lazily.
- `llm.py` wraps the shubiaobiao OpenAI-compatible API for keyword expansion and answering.
- `app.py` exposes the FastAPI endpoints and now serves the static UI in `static/`.
