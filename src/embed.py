"""
embed.py
Generates embeddings for document chunks and persists them in a local
Chroma vector store.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function():
    """Returns the embedding model used across ingestion and retrieval.
    Using a local HuggingFace model keeps embedding free/offline; swap this
    for a watsonx embedding endpoint if you'd rather keep everything on IBM's stack.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks):
    """Embeds chunks and writes/updates the persistent Chroma store."""
    embedding_fn = get_embedding_function()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
    )
    vector_store.persist()
    return vector_store


def load_vector_store():
    """Loads an existing persisted Chroma store."""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"No vector store found at {PERSIST_DIR}. Run ingest.py first."
        )
    embedding_fn = get_embedding_function()
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_fn)
