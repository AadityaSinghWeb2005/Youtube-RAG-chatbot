
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import streamlit as st

from indexing.transcript_loader import load_transcript
from indexing.text_splitter import split_text
from indexing.chroma_db import create_vector_store
from retrieval.retriever import get_retriever
from augmentation.context_and_prompt import create_prompt
from generation.llm import generate_answer


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 YouTube Transcript Chatbot")
st.caption("Ask questions and get answers grounded in a video's transcript.")


# -------------------- HELPER FUNCTIONS --------------------

def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common URL formats."""

    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]

    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.match(
                r"^/(?:embed|shorts|live)/([^/?]+)",
                parsed.path,
            )
            video_id = match.group(1) if match else ""

    else:
        raise ValueError("Please enter a valid YouTube video URL.")

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Could not extract a valid YouTube video ID.")

    return video_id


def initialize_state():
    defaults = {
        "video_id": None,
        "video_url": None,
        "db_path": None,
        "messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# -------------------- VIDEO INPUT --------------------

st.subheader("1. Load a YouTube video")

with st.form("video_form"):
    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    submitted = st.form_submit_button(
        "Load Video",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        video_id = extract_video_id(video_url)
        db_path = str(
            Path("chroma_db") / "videos" / video_id
        )

        # Avoid re-indexing the currently loaded video
        if (
            st.session_state.video_id == video_id
            and st.session_state.db_path == db_path
        ):
            st.info("This video is already loaded.")
        else:
            with st.spinner(
                "Extracting transcript and building the index..."
            ):
                transcript = load_transcript(video_url)

                if not transcript or not transcript.strip():
                    raise ValueError(
                        "The video transcript is empty."
                    )

                chunks = split_text(transcript)

                if not chunks:
                    raise ValueError(
                        "Could not create text chunks."
                    )

                # Store this video's vectors separately
                create_vector_store(
                    chunks,
                    persist_directory=db_path,
                )

                st.session_state.video_id = video_id
                st.session_state.video_url = video_url
                st.session_state.db_path = db_path
                st.session_state.messages = []

            st.success(
                f"Video loaded successfully! "
                f"{len(chunks)} chunks indexed."
            )

    except Exception as e:
        st.error(f"Could not load video: {e}")


# -------------------- CHAT INTERFACE --------------------

if st.session_state.video_id:
    st.divider()

    st.subheader("2. Chat with your video")

    st.markdown(
        f"**Loaded video ID:** `{st.session_state.video_id}`"
    )

    # Display previous conversation messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching the transcript and generating an answer..."
                ):
                    # 1. Retrieval using your existing MMR retriever
                    retriever = get_retriever(
                        persist_directory=st.session_state.db_path,
                        k=4,
                    )

                    documents = retriever.invoke(question)

                    if not documents:
                        answer = (
                            "I couldn't find relevant information "
                            "in this video's transcript."
                        )
                    else:
                        # 2. Augmentation
                        prompt = create_prompt(
                            documents,
                            question,
                        )

                        # 3. Generation using your Ollama Cloud LLM
                        answer = generate_answer(prompt)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info(
        "Enter a YouTube URL above and click Load Video "
        "to start chatting."
    )
