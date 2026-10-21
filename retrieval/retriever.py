
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()


def get_retriever(
    persist_directory: str = "./chroma_db",
    k: int = 4,
):
    embeddings_model = OllamaEmbeddings(
        model="embeddinggemma-2:270m",
        base_url="http://localhost:11434",
    )

    vector_store = Chroma(
        collection_name="youtube_transcripts",
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5},
    )


def retrieve_chunks(question: str) -> list:
    retriever = get_retriever()
    return retriever.invoke(question)


if __name__ == "__main__":
    question = input("Ask a question about the video: ")

    results = retrieve_chunks(question)

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content)
