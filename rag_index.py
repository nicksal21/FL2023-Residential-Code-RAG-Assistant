import json
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

INDEX_DIR = Path(__file__).parent / "vector_index"
EMBEDDING_MODEL = "text-embedding-3-small"


class NumpyVectorIndex:
    def __init__(self, embeddings_matrix, chunks, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = chunks
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        self.normalized = embeddings_matrix / np.clip(norms, 1e-10, None)

    def similarity_search(self, query_vector, k=5):
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(np.linalg.norm(query), 1e-10)
        scores = self.normalized @ query
        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in top_idx]

    def query(self, embeddings_client, question, k=5):
        query_vector = embeddings_client.embed_query(question)
        return self.similarity_search(query_vector, k=k)


def index_available():
    return (
        (INDEX_DIR / "embeddings.npy").is_file()
        and (INDEX_DIR / "chunks.jsonl").is_file()
        and (INDEX_DIR / "meta.json").is_file()
    )


def load_index():
    if not index_available():
        raise FileNotFoundError(
            "Pre-built vector index not found. Run `python build_index.py` locally."
        )

    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    chunks = [
        json.loads(line)
        for line in (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return NumpyVectorIndex(embeddings, chunks, meta.get("embedding_model", EMBEDDING_MODEL))


def make_embeddings_client(api_key):
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
