EconSearch
==========

Minimal PDF ingestion + question answering pipeline focused on the essentials. The project ingests local economics/finance PDFs, builds a lightweight TF-IDF index, and exposes a FastAPI UI for bilingual answers (English + Chinese).

Quick Start (Docker)
--------------------
Docker is the recommended way to try the project because every dependency is bundled in the image.

1. Configure environment variables
   - Copy `.env.example` to `.env` and fill in your LLM key(s).
   - Choose the provider (`shubiaobiao` or `deepseek`) via `LLM_PROVIDER`.
   - Set `DEFAULT_PDF_DIR=/data/pdfs` so the backend knows where the bind-mounted PDFs live.

   Example `.env`:
   ```ini
   LLM_PROVIDER=shubiaobiao
   OPENAI_API_KEY=sk-your-real-key
   SHUBIAOBIAO_BASE_URL=https://api.shubiaobiao.cn/v1/
   SHUBIAOBIAO_MODEL=gpt-4o-mini
   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/
   DEEPSEEK_MODEL=deepseek-chat
   OPENAI_BASE_URL=
   OPENAI_MODEL=
   DEFAULT_PDF_DIR=/data/pdfs
   ```

2. Build the image (only once):
   ```bash
   docker build -t econsearch .
   ```

3. Ingest PDFs inside Docker (PowerShell example, note the backticks ` for line continuation):
   ```powershell
   docker run --rm `
     --env-file .env `
     -v "D:\LLM_Agents\JF_papers\all_pdfs:/data/pdfs" `
     -v "${PWD}\storage:/app/storage" `
     econsearch `
     python ingest.py --pdf-dir /data/pdfs --out storage/paper_index.json
   ```

   Bash/Zsh equivalent:
   ```bash
   docker run --rm      --env-file .env      -v /absolute/path/to/pdfs:/data/pdfs      -v "$(pwd)/storage:/app/storage"      econsearch      python ingest.py --pdf-dir /data/pdfs --out storage/paper_index.json
   ```

4. Run the API/UI (reuse the same volume mounts):
   ```powershell
   docker run --rm `
     --env-file .env `
     -p 8000:8000 `
     -v "D:\LLM_Agents\JF_papers\all_pdfs:/data/pdfs" `
     -v "${PWD}\storage:/app/storage" `
     econsearch
   ```

   ```bash
   docker run --rm      --env-file .env      -p 8000:8000      -v /absolute/path/to/pdfs:/data/pdfs      -v "$(pwd)/storage:/app/storage"      econsearch
   ```

5. Visit `http://localhost:8000/`. Leave the ¡°PDF directory¡± field blank to use the configured `DEFAULT_PDF_DIR`. To ingest another folder temporarily, type its container path (e.g. `/data/other`) before clicking **Ingest**.

Tips:
- `storage/` is mounted so `paper_index.json` persists between container runs.
- To verify the PDFs are visible inside the container: `docker exec -it <container_id> ls /data/pdfs`.
- Rebuild the image (`docker build ...`) after pulling new code.

Local Setup (micromamba / conda)
--------------------------------
Docker is preferred, but you can run locally if you already manage Python environments.

```powershell
micromamba activate econsearch
pip install -r requirements.txt
```

Then configure `.env` (same format as above) and run ingestion:

```powershell
micromamba run -n econsearch python ingest.py --pdf-dir "D:\path\to\pdfs" --out storage/paper_index.json
```

Launch the API locally:

```powershell
micromamba run -n econsearch uvicorn app:app --host 0.0.0.0 --port 8000
```

Available Endpoints
-------------------
- `GET /` : single-page UI for ingestion and bilingual Q&A.
- `GET /health` : readiness probe.
- `GET /info` : JSON summary of available routes.
- `POST /ingest` : trigger ingestion on a directory.
- `POST /ask` : answer a question using retrieved PDFs.
- `POST /reload` : rebuild the in-memory search index from the stored JSON.

Implementation Notes
--------------------
- `ingest.py` extracts per-document metadata (title, authors, keywords, abstract) via `pypdf` and writes a JSON index.
- `search_engine.py` performs TF-IDF search over that metadata and lazily reads full text when the LLM needs context.
- `llm.py` wraps either the shubiaobiao or deepseek OpenAI-compatible API, producing bilingual answers (English + Simplified Chinese) and structured per-paper summaries.
- `app.py` exposes the FastAPI endpoints and serves the static UI (`static/`).
- `storage/` holds the generated `paper_index.json`; mount or back it up for persistent usage.
