
from langchain_core.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions
based on a YouTube video's transcript.

Use only the provided context to answer the user's question.
If the answer is not available in the context, clearly say
that you could not find the answer in the video transcript.

Context:
{context}

Question:
{question}

Instructions:
- Give a clear and accurate answer.
- Do not invent information that is not in the context.
- Use simple language.
"""


def format_context(documents) -> str:
    """Combine retrieved documents into a single context string."""

    if not documents:
        raise ValueError("No retrieved documents were provided.")

    return "\n\n".join(
        f"Source {i}:\n{doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )


def create_prompt(documents, question: str):
    """Build a prompt from retrieved context and the user question."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(documents)

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    return prompt_template.invoke({
        "context": context,
        "question": question,
    })
