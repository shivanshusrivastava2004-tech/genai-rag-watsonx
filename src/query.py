"""
query.py
Main entrypoint: takes a user question, retrieves relevant context,
builds a guarded prompt, and calls IBM watsonx to generate an answer.
"""

import argparse
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from retriever import get_relevant_chunks, format_context
from prompts import build_prompt

load_dotenv()

MODEL_ID = "ibm/granite-13b-instruct-v2"  # swap for any watsonx-hosted model you have access to


def get_watsonx_model():
    credentials = Credentials(
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        api_key=os.getenv("WATSONX_API_KEY"),
    )
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not credentials.api_key or not project_id:
        raise EnvironmentError(
            "Missing WATSONX_API_KEY or WATSONX_PROJECT_ID. Copy .env.example to .env and fill it in."
        )

    return ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=project_id,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "temperature": 0.2,
        },
    )


def answer_question(question: str, k: int = 4) -> str:
    chunks = get_relevant_chunks(question, k=k)
    if not chunks:
        return "No relevant documents found. Run ingest.py first to populate the vector store."

    context = format_context(chunks)
    prompt = build_prompt(context, question)

    model = get_watsonx_model()
    response = model.generate_text(prompt=prompt)
    return response


def main():
    parser = argparse.ArgumentParser(description="Query the RAG-powered document Q&A assistant.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    answer = answer_question(args.question, k=args.k)
    print("\nANSWER:\n" + answer)


if __name__ == "__main__":
    main()
