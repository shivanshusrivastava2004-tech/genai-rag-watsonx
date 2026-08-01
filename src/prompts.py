"""
prompts.py
Prompt templates used to ground the LLM's answers in retrieved context
and reduce hallucination.
"""

SYSTEM_GUARDRAILS = """You are a document Q&A assistant. Follow these rules strictly:
1. Only answer using the information provided in the CONTEXT below.
2. If the answer is not present in the context, say "I don't have enough information in the provided documents to answer that."
3. Do not use outside knowledge or make assumptions beyond the context.
4. Keep answers concise and cite which part of the context supports your answer when relevant.
"""

FEW_SHOT_EXAMPLES = """
Example 1:
CONTEXT: "The system achieved a 94% true positive rate during testing."
QUESTION: What was the true positive rate?
ANSWER: The true positive rate was 94%, based on the testing results described in the document.

Example 2:
CONTEXT: "The report discusses cloud security but does not mention pricing."
QUESTION: What is the cost of the solution?
ANSWER: I don't have enough information in the provided documents to answer that.
"""

QA_PROMPT_TEMPLATE = """{system_guardrails}

{few_shot_examples}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_prompt(context: str, question: str) -> str:
    """Assembles the final prompt sent to the watsonx LLM."""
    return QA_PROMPT_TEMPLATE.format(
        system_guardrails=SYSTEM_GUARDRAILS,
        few_shot_examples=FEW_SHOT_EXAMPLES,
        context=context,
        question=question,
    )
