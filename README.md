# Construction Document RAG Pipeline

A Flask demo of a retrieval-augmented generation (RAG) pipeline for construction-domain Q&A. Built to explore LangChain, OpenAI embeddings, and vector search.

This project was originally prototyped against Florida residential building code PDFs. **Those documents are not included or redistributed** — publishing them would violate copyright. The public repo instead ships **original sample markdown** in `sample_docs/` so the architecture remains demonstrable without proprietary content.

## Architecture

- **Python** — Flask, LangChain, OpenAI
- **Ingestion** — Markdown header-based chunking, OpenAI `text-embedding-3-large`
- **Retrieval** — In-memory vector store, top-k similarity search
- **Generation** — GPT with context-grounded system instructions
- **Frontend** — HTML/CSS Flask templates

## How it works

1. `sample_docs/` contains short, original reference material (foundations, framing, electrical/safety).
2. On startup, `app.py` loads and chunks the markdown by headers.
3. Chunks are embedded and stored in an in-memory vector store.
4. A user submits a question via POST.
5. The top-k similar chunks are retrieved and passed to the model as context.
6. The model answers using only that context and cites sources when possible.

## Optional: parse your own PDFs locally

`doc_parser.py` is a **local-only** utility using [Docling](https://github.com/docling-project/docling). Place your own licensed PDFs in `local_pdfs/` (gitignored), run the script, and review output in `local_mds/`. Do not commit copyrighted material.

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

Create a `.env` file (never commit this):

```env
OPENAI_API_KEY=your-key-here
FLASK_KEY=your-secret-key-here
```

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

## Deploy (e.g. Render)

The portfolio embeds this app at `https://construction-rag-pipeline.onrender.com` once deployed.

1. Push this repo to GitHub.
2. In [Render](https://render.com), create a **Blueprint** from `render.yaml` or a **Web Service** with start command `gunicorn app:app`.
3. Set `OPENAI_API_KEY` in the Render dashboard (`FLASK_KEY` can be auto-generated).
4. After deploy, confirm the app loads and the portfolio iframe works on `https://nicksal21.github.io`.

The app sets `Content-Security-Policy: frame-ancestors` for GitHub Pages and uses cross-site session cookies in production so form submissions work inside the embedded iframe.

```bash
gunicorn app:app
```

A `Procfile` is included for platforms that use it.

## Copyright note

Official building codes are copyrighted publications. This repository:

- Does **not** distribute code books, PDFs, or converted markdown from those sources
- Uses **original sample content** for the live demo
- Documents how you can run `doc_parser.py` on documents **you are licensed to use**

For real compliance work, consult the authority having jurisdiction and licensed professionals.

## Ideas for extension

- Bring-your-own-document upload (user supplies text; nothing redistributed)
- Persistent vector store (Chroma, Pinecone) instead of in-memory
- Chat history and follow-up questions
- Swap sample_docs for your own licensed corpus in a private deployment
