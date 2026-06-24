from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import sys
from os import listdir
from os.path import isfile, join

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from flask import Flask, jsonify, request, render_template, redirect, url_for, session

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
RETRIEVAL_K = 5

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_KEY", "dev-only-change-in-production")

IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or bool(os.getenv("RENDER"))
app.config.update(
    SESSION_COOKIE_SAMESITE="None" if IS_PRODUCTION else "Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

vector_store = None
openai_client = None
_openai_client_key = None


@app.after_request
def set_embed_headers(response):
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://nicksal21.github.io https://*.github.io "
        "http://localhost:* http://127.0.0.1:*"
    )
    response.headers.pop("X-Frame-Options", None)
    return response


def resolve_api_key(form_key=None):
    load_dotenv()
    if form_key and form_key.strip():
        return form_key.strip()
    if session.get("openai_api_key"):
        return session["openai_api_key"]
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return None


def create_vector_store(api_key):
    if not SAMPLE_DOCS_DIR.is_dir():
        raise FileNotFoundError(
            f"Sample docs directory not found: {SAMPLE_DOCS_DIR}"
        )

    file_paths = sorted(
        f for f in listdir(SAMPLE_DOCS_DIR) if isfile(join(SAMPLE_DOCS_DIR, f))
    )
    if not file_paths:
        raise FileNotFoundError(f"No markdown files found in {SAMPLE_DOCS_DIR}")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    docs = []
    for file_name in file_paths:
        file_path = SAMPLE_DOCS_DIR / file_name
        content = file_path.read_text(encoding="utf-8")
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        for doc in md_header_splits:
            doc.metadata["source"] = file_name
        docs.extend(md_header_splits)

    embed = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    store = InMemoryVectorStore(embedding=embed)
    store.add_documents(docs)
    return store


def ensure_ready(api_key):
    global vector_store, openai_client, _openai_client_key

    if vector_store is None:
        vector_store = create_vector_store(api_key)

    if openai_client is None or _openai_client_key != api_key:
        openai_client = OpenAI(api_key=api_key)
        _openai_client_key = api_key


def ask_question(store, client, question, k=RETRIEVAL_K):
    results = store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(
        f"Source: {result.metadata.get('source')}\n"
        f"Headers: {result.metadata}\n\n"
        f"Content: {result.page_content}"
        for result in results
    )

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You answer questions using only the provided construction reference context. "
            "The context is sample educational material, not an official building code. "
            "If the context does not contain the answer, say you do not know. "
            "Cite the source filename and relevant section/header when possible. "
            "Remind the user to consult licensed professionals and local codes for real projects."
        ),
        input=f"""
Question:
{question}

Context:
{context}
        """,
    )

    return response.output_text


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def home():
    answer = session.pop("answer", None)
    error = session.pop("error", None)
    return render_template(
        "index.html",
        answer=answer,
        error=error,
        has_api_key=bool(resolve_api_key()),
    )


@app.route("/submit", methods=["POST"])
def handle_data():
    user_input = request.form.get("user_input", "").strip()
    api_key_input = request.form.get("openai_api_key", "").strip()

    if api_key_input:
        session["openai_api_key"] = api_key_input

    api_key = resolve_api_key(api_key_input)
    if not api_key:
        session["error"] = (
            "Add your OpenAI API key to use this demo. "
            "Your key is stored only in this browser session."
        )
        return redirect(url_for("home"))

    if not user_input:
        return redirect(url_for("home"))

    try:
        ensure_ready(api_key)
        answer = ask_question(vector_store, openai_client, user_input)
        session["answer"] = answer
    except Exception:
        session["error"] = (
            "Could not process your question. Check your API key and try again."
        )

    return redirect(url_for("home"))


if __name__ == "__main__":
    if not resolve_api_key() and sys.stdin.isatty():
        print(
            "No OPENAI_API_KEY found. You can set it in .env or enter it in the web form.",
            file=sys.stderr,
        )
    app.run(debug=True)
