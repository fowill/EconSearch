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
DEFAULT_PDF_DIR=/data/pdfs
```

Set `LLM_PROVIDER` to `deepseek` if you want to switch. Restart the app after editing `.env` so the new values are picked up.

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

Docker
------
The container workflow lets you ship or reproduce the project quickly. Below is a complete walkthrough assuming Docker Desktop is installed.

### 1. Prepare configuration

1. Copy `.env.example` to `.env` and fill in your keys.
2. For container usage set `DEFAULT_PDF_DIR` to the path inside the container (the guide below uses `/data/pdfs`).
3. Ensure your local PDF directory exists; we will bind mount it later.

### 2. Build the image

```bash
docker build -t econsearch .
```

### 3. One-off ingestion inside Docker

Mount your host PDF folder to `/data/pdfs` and the `storage/` folder so the generated index persists:

```powershell
# PowerShell syntax (use backticks for line continuation)
docker run --rm `
  --env-file .env `
  -v "D:\LLM_Agents\JF_papers\all_pdfs:/data/pdfs" `
  -v "${PWD}\storage:/app/storage" `
  econsearch `
  python ingest.py --pdf-dir /data/pdfs --out storage/paper_index.json
```

```bash
# Bash/Zsh syntax
docker run --rm \
  --env-file .env \
  -v /absolute/path/to/pdfs:/data/pdfs \
  -v "$(pwd)/storage:/app/storage" \
  econsearch \
  python ingest.py --pdf-dir /data/pdfs --out storage/paper_index.json
```

### 4. Run the API/UI service

Re-use the same mounts when starting the web app. With `DEFAULT_PDF_DIR=/data/pdfs` configured you can leave the PDF field blank in the UI and the backend will use the mounted directory.

```powershell
docker run --rm `
  --env-file .env `
  -p 8000:8000 `
  -v "D:\LLM_Agents\JF_papers\all_pdfs:/data/pdfs" `
  -v "${PWD}\storage:/app/storage" `
  econsearch
```

```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  -v /absolute/path/to/pdfs:/data/pdfs \
  -v "$(pwd)/storage:/app/storage" \
  econsearch
```

Now browse to `http://localhost:8000/`.

### 5. Tips

- `storage/` keeps the `paper_index.json` persistent. You can mount another location if desired.
- If you want to ingest from a different folder temporarily, type the container path (e.g. `/data/other`) into the UI before clicking **Ingest**; Windows paths will not work inside Docker.
- To check the mount inside a running container: `docker exec -it <container_id> ls /data/pdfs`.

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
- `llm.py` wraps the configured OpenAI-compatible API for keyword expansion and answering.
- `app.py` exposes the FastAPI endpoints and now serves the static UI in `static/`.
