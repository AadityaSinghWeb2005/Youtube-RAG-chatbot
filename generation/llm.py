
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def generate_answer(prompt) -> str:
    """Send the augmented prompt to Ollama Cloud and return the answer."""

    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        raise ValueError("OLLAMA_API_KEY not found in .env")

    llm = ChatOpenAI(
        model="gpt-oss:120b",
        base_url="https://ollama.com/v1",
        api_key=api_key,
        temperature=0.2,
    )

    response = llm.invoke(prompt)

    return response.content
