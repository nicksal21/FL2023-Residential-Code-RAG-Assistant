from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import getpass
import os
from os import listdir
from os.path import isfile, join

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from flask import Flask, request, render_template, redirect, url_for, session

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


@app.after_request
def set_embed_headers(response):
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://nicksal21.github.io https://*.github.io "
        "http://localhost:* http://127.0.0.1:*"
    )
    response.headers.pop("X-Frame-Options", None)
    return response


def load_environment_variables():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key


def create_vector_store():
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

    embed = OpenAIEmbeddings(model="text-embedding-3-large")
    store = InMemoryVectorStore(embedding=embed)
    store.add_documents(docs)
    return store


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


def init_app():
    global vector_store, openai_client
    load_environment_variables()
    openai_client = OpenAI()
    vector_store = create_vector_store()


@app.route("/")
def home():
    answer = session.pop("answer", None)
    return render_template("index.html", answer=answer)


@app.route("/submit", methods=["POST"])
def handle_data():
    user_input = request.form.get("user_input", "").strip()
    if not user_input:
        return redirect(url_for("home"))

    answer = ask_question(vector_store, openai_client, user_input)
    session["answer"] = answer
    return redirect(url_for("home"))


init_app()

if __name__ == "__main__":
    app.run(debug=True)
