"""
retriever.py
Wraps the vector store to fetch the top-k most relevant chunks for a query.
"""

from embed import load_vector_store


def get_relevant_chunks(query: str, k: int = 4):
    """Returns the top-k document chunks most semantically similar to the query."""
    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    # results: list of (Document, score) tuples, lower score = more similar (distance)
    chunks = [doc for doc, _score in results]
    return chunks


def format_context(chunks) -> str:
    """Joins retrieved chunks into a single context block for the prompt."""
    return "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
