from langchain_community.document_loaders import YoutubeLoader


def load_transcript(video_url: str) -> str:
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=False,
        language=["en", "hi"],
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No transcript found.")

    return documents[0].page_content