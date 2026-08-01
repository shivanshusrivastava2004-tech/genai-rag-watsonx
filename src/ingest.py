"""
ingest.py
Loads documents from a folder, splits them into chunks, and prepares them
for embedding. Supports .pdf and .txt files.
"""

import argparse
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(folder_path: str):
    """Load all .pdf and .txt files from a folder into LangChain Document objects."""
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
        else:
            print(f"Skipping unsupported file: {filename}")
    return documents


def chunk_documents(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents for the RAG pipeline.")
    parser.add_argument("--docs", type=str, required=True, help="Path to folder containing documents.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading documents from: {args.docs}")
    docs = load_documents(args.docs)
    print(f"Loaded {len(docs)} raw document(s).")

    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunk(s).")

    # Hand off to embed.py's build_vector_store() to persist these chunks
    from embed import build_vector_store
    build_vector_store(chunks)
    print("Ingestion complete. Vector store updated.")


if __name__ == "__main__":
    main()
