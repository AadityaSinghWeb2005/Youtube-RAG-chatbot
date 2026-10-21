from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def create_vector_store(
    chunks: list[str],
    persist_directory: str = "./chroma_db",
) -> Chroma:

    if not chunks:
        raise ValueError("Chunks list cannot be empty.")

    embeddings_model = OllamaEmbeddings(
        model="embeddinggemma-2:270m",
        base_url="http://localhost:11434",
    )

    vector_store = Chroma(
        collection_name="youtube_transcripts",
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    vector_store.add_texts(
        texts=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vector_store