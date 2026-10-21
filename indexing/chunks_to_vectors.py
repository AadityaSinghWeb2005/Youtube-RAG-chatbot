import ollama


def chunks_to_embeddings(chunks: list[str]) -> list[list[float]]:
    if not chunks:
        raise ValueError("Chunks list cannot be empty.")

    response = ollama.embed(
        model="embeddinggemma",
        input=chunks,
    )

    vectors = response["embeddings"]

    print(f"Generated {len(vectors)} embeddings.")
    print(f"Vector dimensions: {len(vectors[0])}")

    return vectors