from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    return splitter.split_text(transcript)