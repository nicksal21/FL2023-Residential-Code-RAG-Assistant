from pathlib import Path
import json

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from os import listdir
from os.path import isfile, join

DOCS_DIR = Path(__file__).parent / "florida_residential_code_mds"
INDEX_DIR = Path(__file__).parent / "vector_index"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64


def load_chunks():
    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {DOCS_DIR}")

    file_paths = sorted(
        f for f in listdir(DOCS_DIR) if isfile(join(DOCS_DIR, f)) and f.endswith(".md")
    )
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    chunks = []
    for file_name in file_paths:
        content = (DOCS_DIR / file_name).read_text(encoding="utf-8")
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        for doc in splitter.split_text(content):
            chunks.append(
                {
                    "page_content": doc.page_content,
                    "metadata": {**doc.metadata, "source": file_name},
                }
            )
    return chunks


def build_index(api_key):
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No chunks found to index.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    vectors = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_vectors = embeddings.embed_documents(
            [item["page_content"] for item in batch]
        )
        vectors.extend(batch_vectors)
        print(f"Embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    matrix = np.asarray(vectors, dtype=np.float32)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", matrix)

    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote index with {len(chunks)} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env before running build_index.py")
    build_index(api_key)
