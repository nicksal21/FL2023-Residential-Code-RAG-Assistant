# Construction Document RAG Pipeline

A Flask demo of a retrieval-augmented generation (RAG) pipeline for Florida Residential Building Code Q&A. Built to explore LangChain, OpenAI embeddings, and vector search.

Official code **PDFs are not included** in this repository. The app indexes **markdown chapter files** in `florida_residential_code_mds/` that were converted locally with Docling. Some tables and figures may be missing or imperfect in the conversion.

## Architecture

- **Python** — Flask, LangChain, OpenAI
- **Ingestion** — Markdown header-based chunking, OpenAI `text-embedding-3-large`
- **Retrieval** — In-memory vector store, top-k similarity search
- **Generation** — GPT with context-grounded system instructions
- **Frontend** — HTML/CSS Flask templates

## How it works

1. `florida_residential_code_mds/` contains 46 chapter markdown files from the 2023 Florida Residential Building Code.
2. `build_index.py` embeds those chapters and writes a compact `vector_index/` used at runtime.
3. Render loads the pre-built index instead of embedding 3,000+ chunks on each deploy (fits the 512 MB free tier).
4. A user submits a question with their own OpenAI API key (BYOK).
5. The top-k similar chunks are retrieved and passed to the model as context.

### Rebuild the search index

After changing markdown files, rebuild locally and commit `vector_index/`:

```bash
export OPENAI_API_KEY=your-key-here
python build_index.py
```

## Optional: re-parse PDFs locally

`doc_parser.py` is a **local-only** utility using [Docling](https://github.com/docling-project/docling). Place licensed PDFs in `local_pdfs/` (gitignored), run the script, and review output in `local_mds/`.

```bash
pip install docling
mkdir local_pdfs
# add your PDFs, then:
python doc_parser.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: create a `.env` file for local development (never commit this):

```env
OPENAI_API_KEY=your-key-here
FLASK_KEY=your-secret-key-here
```

If `OPENAI_API_KEY` is not set locally, enter your key in the web form instead.

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

## Deploy (e.g. Render)

Live demo: https://fl2023-residential-code-rag-assistant.onrender.com

**You do not need to put your OpenAI API key in Render** unless you want the server to use a shared key.

By default, the hosted demo uses **bring-your-own-key (BYOK)**:

- The app starts without `OPENAI_API_KEY`
- Each visitor enters their own key in the browser
- The key is kept in the Flask session for that visit only

To deploy:

1. Push this repo to GitHub.
2. In [Render](https://render.com), create a **Web Service** with start command `gunicorn app:app`.
3. Set `FLASK_KEY` (or let Render generate it). `OPENAI_API_KEY` is optional.
4. Confirm `https://your-service.onrender.com/health` returns `{"status":"ok"}`.

The app sets `Content-Security-Policy: frame-ancestors` for GitHub Pages and uses cross-site session cookies in production so form submissions work inside the embedded iframe.

```bash
gunicorn app:app
```

A `Procfile` and `render.yaml` are included for platforms that use them.

## Copyright note

Official building code PDFs are not distributed in this repository. The committed markdown is text converted for a technical demo. For compliance work, consult the authority having jurisdiction, licensed professionals, and the official published code.

## Ideas for extension

- Improve Docling post-processing for tables and figure references
- Persistent vector store (Chroma, Pinecone) instead of in-memory
- Chat history and follow-up questions
- Precompute embeddings at build time to reduce first-query latency
