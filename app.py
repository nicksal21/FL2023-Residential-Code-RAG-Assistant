from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

from flask import Flask, jsonify, request, render_template, redirect, url_for, session

from rag_index import index_available, load_index, make_embeddings_client

RETRIEVAL_K = 5

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_KEY", "dev-only-change-in-production")

IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or bool(os.getenv("RENDER"))
app.config.update(
    SESSION_COOKIE_SAMESITE="None" if IS_PRODUCTION else "Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

vector_index = None
embeddings_client = None
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
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return None


def ensure_ready(api_key):
    global vector_index, embeddings_client, openai_client, _openai_client_key

    if vector_index is None:
        vector_index = load_index()

    if embeddings_client is None or _openai_client_key != api_key:
        embeddings_client = make_embeddings_client(api_key)
        openai_client = OpenAI(api_key=api_key)
        _openai_client_key = api_key


def ask_question(index, embed_client, client, question, k=RETRIEVAL_K):
    results = index.query(embed_client, question, k=k)
    context = "\n\n---\n\n".join(
        f"Source: {result['metadata'].get('source')}\n"
        f"Headers: {result['metadata']}\n\n"
        f"Content: {result['page_content']}"
        for result in results
    )

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You answer questions using only the provided Florida Residential Code context. "
            "The source markdown was converted from code publications and may omit some tables "
            "or figures. If the context does not contain the answer, say you do not know. "
            "Cite the source filename and relevant section/header when possible. "
            "Remind the user to consult licensed professionals and the official code for compliance."
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
    return jsonify(
        {
            "status": "ok",
            "index_ready": index_available(),
        }
    ), 200


@app.route("/")
def home():
    answer = session.pop("answer", None)
    error = session.pop("error", None)
    if not index_available() and not error:
        error = "Search index is missing. Redeploy after running build_index.py."
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

    api_key = resolve_api_key(api_key_input)
    if not api_key:
        session["error"] = (
            "Add your OpenAI API key to use this demo. "
            "Your key is kept in this browser tab only while the page is open."
        )
        return redirect(url_for("home"))

    if not user_input:
        return redirect(url_for("home"))

    if not index_available():
        session["error"] = "Search index is unavailable. Please try again later."
        return redirect(url_for("home"))

    try:
        ensure_ready(api_key)
        answer = ask_question(vector_index, embeddings_client, openai_client, user_input)
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
