from indexing.transcript_loader import load_transcript
from indexing.text_splitter import split_text
from indexing.chunks_to_vectors import chunks_to_embeddings
from indexing.chroma_db import create_vector_store
from retrieval.retriever import retrieve_chunks
from augmentation.context_and_prompt import create_prompt
from generation.llm import generate_answer

#Hi
def main():
    video_url = input("Enter YouTube video URL: ")

    # Step 1: Extract transcript
    transcript = load_transcript(video_url)
    

    # Step 2: Convert transcript into chunks
    chunks = split_text(transcript)
    

    # Step 3: Convert chunks into embeddings
    embeddings = chunks_to_embeddings(chunks)

    # Step 4: Store chunks and embeddings in ChromaDB
    db = create_vector_store(chunks)


    #step-5: Retrieve relevant chunks based on a user query
    question = input("Ask a question about the video: ")
    documents = retrieve_chunks(question)

    #step-6: Create a prompt using the retrieved context and user question
    prompt = create_prompt(documents, question)

    #step-7 : Generate an answer using the LLM
    answer = generate_answer(prompt)

    print(f"Transcript length: {len(transcript)} characters")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embeddings generated for {len(embeddings)} chunks.")
    print(f"Retrieved {len(documents)} relevant chunks.")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()