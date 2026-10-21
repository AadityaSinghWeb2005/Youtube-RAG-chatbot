# 🎬 YT-Notes — YouTube Transcript Chatbot

A **Retrieval-Augmented Generation (RAG)** application that lets you *chat with any YouTube video*. Paste a video URL, let the app extract and index the video's transcript, then ask questions and get answers **grounded only in what was actually said in the video**.

The project ships with two front-ends that share the same RAG pipeline:

| Entry point | Interface | Best for |
|---|---|---|
| `app.py` | Streamlit web app | Interactive chat with conversation history |
| `main.py` | Terminal (CLI) | Quick one-off experiments and learning the pipeline |

---

## ✨ Features

- **Automatic transcript extraction** — pulls captions directly from YouTube (English and Hindi supported), no manual copying.
- **Flexible URL parsing** — understands `watch?v=`, `youtu.be`, `/embed/`, `/shorts/`, `/live/`, mobile, and YouTube Music links, and validates the 11-character video ID.
- **RAG pipeline** — chunk → embed → store → retrieve → augment → generate, built on LangChain.
- **Smart retrieval (MMR)** — Maximal Marginal Relevance search returns *diverse and relevant* chunks instead of near-duplicates.
- **Grounded answers** — the prompt strictly instructs the LLM to answer *only* from the transcript and to admit when the answer isn't there, reducing hallucination.
- **Persistent vector indexes** — every video gets its own on-disk ChromaDB store (`chroma_db/videos/<video_id>`), so a video is indexed once and reused.
- **Chat memory** — the Streamlit app keeps the full conversation in session state and re-renders history.
- **Dual execution modes** — a polished web UI and a minimal, step-by-step CLI.

---

## 🏗️ How It Works (Architecture)

```
                        ┌─────────────────────────────────────────────┐
                        │              YouTube Video URL              │
                        └──────────────────────┬──────────────────────┘
                                               │
              ┌────────────────────────────────▼────────────────────────────────┐
              │ 1. LOAD          indexing/transcript_loader.py                  │
              │    Fetch the transcript via LangChain's YoutubeLoader             │
              └────────────────────────────────┬────────────────────────────────┘
                                               │
              ┌────────────────────────────────▼────────────────────────────────┐
              │ 2. SPLIT         indexing/text_splitter.py                      │
              │    RecursiveCharacterTextSplitter (1000 chars / 200 overlap)      │
              └────────────────────────────────┬────────────────────────────────┘
                                               │
              ┌────────────────────────────────▼────────────────────────────────┐
              │ 3. EMBED & 4. STORE   indexing/chroma_db.py                     │
              │    Ollama embeddings (embeddinggemma-2:270m) → ChromaDB           │
              │    persisted at chroma_db/videos/<video_id>                       │
              └────────────────────────────────┬────────────────────────────────┘
                                               │
        ┌──────────────────────────────────────┘  User asks a question…
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. RETRIEVE       retrieval/retriever.py                     │
│    MMR similarity search over the video's chunks (k = 4)     │
└──────────────────────────────────────┬───────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────┐
│ 6. AUGMENT        augmentation/context_and_prompt.py         │
│    Format retrieved chunks as numbered sources and wrap      │
│    them with the question in a grounding prompt template     │
└──────────────────────────────────────┬───────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────┐
│ 7. GENERATE       generation/llm.py                          │
│    Ollama Cloud LLM (gpt-oss:120b, temperature 0.2)          │
│    produces the final transcript-grounded answer             │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Yt-Notes/
├── app.py                          # Streamlit web app (chat interface)
├── main.py                         # CLI version of the pipeline
├── transcript.txt                  # Sample transcript kept for reference
├── requirement.txt                 # Python dependencies
│
├── indexing/                       # ── Step 1–4: build the knowledge base ──
│   ├── transcript_loader.py        #   Fetch transcript from YouTube
│   ├── text_splitter.py            #   Chunk the transcript
│   ├── chunks_to_vectors.py        #   Embed chunks via the `ollama` client (standalone helper)
│   ├── ollama_cloud_embeddings.py  #   Custom LangChain Embeddings class for the Ollama Cloud API
│   └── chroma_db.py                #   Create & persist the ChromaDB vector store
│
├── retrieval/                      # ── Step 5: find relevant context ──
│   └── retriever.py                #   MMR retriever over ChromaDB (+ standalone test mode)
│
├── augmentation/                   # ── Step 6: build the prompt ──
│   └── context_and_prompt.py       #   Context formatting + grounding prompt template
│
├── generation/                     # ── Step 7: produce the answer ──
│   └── llm.py                      #   LLM call to Ollama Cloud
│
└── chroma_db/                      # Persisted vector stores (gitignored, created at runtime)
    ├── videos/<video_id>/          # One index per video
    └── …                           # Default store used by the CLI
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Transcript fetching | `youtube-transcript-api` (via `langchain-community` `YoutubeLoader`) | Download video captions |
| Orchestration | `langchain`, `langchain-core` | Pipeline building blocks, prompt templates |
| Chunking | `langchain-text-splitters` | Recursive character splitting |
| Embeddings | `ollama` / `langchain-ollama` (`embeddinggemma-2:270m`) | Convert text into vectors |
| Vector store | `chromadb` + `langchain-chroma` | Persistent similarity-search database |
| LLM | `langchain-openai` client → **Ollama Cloud** (`gpt-oss:120b`) | Answer generation |
| Web UI | `streamlit` | Chat interface with session state |
| Config | `python-dotenv` | Load `OLLAMA_API_KEY` from `.env` |
| HTTP | `requests` | Direct calls to the Ollama Cloud embed endpoint |
| Runtime | Python **3.12** (virtual env in `.venv/`) | — |

---

## 📋 Prerequisites

1. **Python 3.12+** — the project is developed and tested on 3.12.
2. **An Ollama Cloud account & API key** — the LLM runs on Ollama Cloud, not your machine. Get a key at [https://ollama.com](https://ollama.com).
3. **A local Ollama server** *(optional, embeddings only)* — the default embedding path uses a local Ollama instance at `http://localhost:11434` with the `embeddinggemma-2:270m` model:
   ```bash
   ollama pull embeddinggemma:270m
   ollama serve   # usually already running as a service
   ```
   If you prefer not to run local embeddings, `indexing/ollama_cloud_embeddings.py` provides a ready-made `Embeddings` class that calls the hosted `https://ollama.com/api/embed` endpoint with your API key instead.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Yt-Notes

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirement.txt
```

> **Note:** `requirement.txt` lists the direct dependencies. The pipeline additionally relies on the LangChain integrations installed alongside them (`langchain-chroma`, `langchain-ollama`, `langchain-openai`), which come in as part of the LangChain package family.

### Configuration

Create a `.env` file in the project root (already gitignored) containing your Ollama Cloud key:

```env
OLLAMA_API_KEY=your-ollama-cloud-api-key
```

`generation/llm.py` loads this key at call time and raises a clear error if it's missing.

---

## 🚀 Usage

### Web App (recommended)

```bash
streamlit run app.py
```

Then in your browser:

1. **Load a video** — paste a YouTube URL and click **Load Video**. The app extracts the transcript, chunks it, and builds a dedicated ChromaDB index at `chroma_db/videos/<video_id>`.
2. **Chat** — type a question in the chat box. Each message runs retrieve → augment → generate, and the answer appears in the chat thread with full history preserved in the session.

The app reuses the in-session index when you re-submit the same video URL, and shows friendly errors for invalid URLs, missing transcripts, or empty indexes.

### CLI

```bash
python main.py
```

The CLI walks through the entire pipeline interactively:

```
Enter YouTube video URL: https://www.youtube.com/watch?v=LPZh9BOjkQs
Ask a question about the video: What is a vector?
Transcript length: 5412 characters
Total chunks: 6
Embeddings generated for 6 chunks.
Retrieved 4 relevant chunks.
Answer: A vector is defined by its length and direction…
```

Two additional standalone test modes are handy for debugging individual stages:

- `python -m retrieval.retriever` — type a question and print the retrieved raw chunks from the default store (`./chroma_db`).
- `python -m indexing.chunks_to_vectors` (or import it) — verify embedding output counts and dimensions.

---

## 🔗 Supported YouTube URL Formats

| Format | Example |
|---|---|
| Standard watch link | `https://www.youtube.com/watch?v=<id>` |
| Short link | `https://youtu.be/<id>` |
| Embed link | `https://www.youtube.com/embed/<id>` |
| Shorts | `https://www.youtube.com/shorts/<id>` |
| Live | `https://www.youtube.com/live/<id>` |
| Mobile | `https://m.youtube.com/watch?v=<id>` |
| YouTube Music | `https://music.youtube.com/watch?v=<id>` |

Anything else raises `ValueError("Please enter a valid YouTube video URL.")`, and the video ID is validated against the standard 11-character `[A-Za-z0-9_-]{11}` pattern. Query parameters like `&t=` are ignored safely.

---

## 📦 Module Reference

### `indexing/`

| Module | What it does |
|---|---|
| `transcript_loader.py` | `load_transcript(video_url)` — builds a `YoutubeLoader` requested in English, falling back to Hindi (`language=["en", "hi"]`). Returns the transcript as a single string; raises `ValueError` if no transcript exists. |
| `text_splitter.py` | `split_text(transcript)` — splits with a `RecursiveCharacterTextSplitter` using **chunk size 1000** and **overlap 200** characters, recursively honoring natural boundaries (paragraphs, sentences). |
| `chunks_to_vectors.py` | `chunks_to_embeddings(chunks)` — standalone embedding helper using the native `ollama` client (`ollama.embed`) with model `embeddinggemma`; returns vectors and prints their count/dimensions. |
| `ollama_cloud_embeddings.py` | `OllamaCloudEmbeddings` — a custom `langchain_core.embeddings.Embeddings` implementation that POSTs to `https://ollama.com/api/embed` with a Bearer token (120 s timeout), implementing both `embed_documents` and `embed_query`. Lets the whole pipeline run without a local Ollama server. |
| `chroma_db.py` | `create_vector_store(chunks, persist_directory="./chroma_db")` — embeds via `OllamaEmbeddings(model="embeddinggemma-2:270m", base_url="http://localhost:11434")` and stores every chunk as `chunk_0`, `chunk_1`, … in a collection named `youtube_transcripts`, persisted under `chroma_db/videos/<video_id>` when called from the web app. |

### `retrieval/`

| Module | What it does |
|---|---|
| `retriever.py` | `get_retriever(persist_directory="./chroma_db", k=4)` — reopens a persisted Chroma store and returns an **MMR retriever** (`search_type="mmr"`, `k=4`, `lambda_mult=0.5`, which balances relevance against diversity). `retrieve_chunks(question)` is a thin convenience wrapper. Running the module directly offers an interactive chunk-retrieval REPL. |

### `augmentation/`

| Module | What it does |
|---|---|
| `context_and_prompt.py` | `create_prompt(documents, question)` — concatenates retrieved chunks as `Source 1: … / Source 2: …` and fills a `ChatPromptTemplate` whose instructions enforce **context-only answering**, explicit "not found in the transcript" fallbacks, no invented facts, and simple language. |

### `generation/`

| Module | What it does |
|---|---|
| `llm.py` | `generate_answer(prompt)` — reads `OLLAMA_API_KEY`, points a `ChatOpenAI` client at the Ollama Cloud OpenAI-compatible endpoint (`https://ollama.com/v1`) with model `gpt-oss:120b` and `temperature=0.2` (low randomness for factual answers), and returns the answer text. |

---

## 🗃️ Data Persistence

- Vector stores live on disk in **`chroma_db/`** (gitignored):
  - `chroma_db/videos/<video_id>/` — one isolated index per video, created by the web app.
  - `chroma_db/` (root) — the default store used by the CLI and the standalone retriever test.
- Collection name: `youtube_transcripts`.
- Each chunk is stored with a deterministic ID (`chunk_0`, `chunk_1`, …) inside its own per-video database, so re-indexing the same video building fresh IDs is safe and isolated per directory.
- Because indexes persist, a previously loaded video is available again in later sessions of the CLI retriever without re-extracting captions (loading a video in the web app still triggers a fresh index build for that video's directory).

---

## 🧪 RAG Configuration at a Glance

| Setting | Value | Where |
|---|---|---|
| Chunk size | 1000 characters | `indexing/text_splitter.py` |
| Chunk overlap | 200 characters | `indexing/text_splitter.py` |
| Embedding model | `embeddinggemma-2:270m` | `indexing/chroma_db.py`, `retrieval/retriever.py` |
| Embedding server | `http://localhost:11434` (local Ollama) | same |
| Vector store | ChromaDB, collection `youtube_transcripts` | same |
| Retrieval strategy | MMR, `k=4`, `lambda_mult=0.5` | `retrieval/retriever.py` |
| LLM model | `gpt-oss:120b` via Ollama Cloud | `generation/llm.py` |
| Temperature | 0.2 | `generation/llm.py` |
| Transcript languages | English → Hindi fallback | `indexing/transcript_loader.py` |
| Answer grounding | Context-only prompt with "not found" fallback | `augmentation/context_and_prompt.py` |

---

## 🐞 Troubleshooting

| Problem / Message | Cause & Fix |
|---|---|
| `OLLAMA_API_KEY not found in .env` | The `.env` file is missing or the key wasn't set. Create `.env` in the project root with `OLLAMA_API_KEY=…`. |
| `No transcript found.` | The video has no captions (auto or manual) in the supported languages — try a different video, or another language. |
| `Please enter a valid YouTube video URL.` / `Could not extract a valid YouTube video ID.` | The URL isn't a recognized YouTube format. Use one of the supported formats above. |
| `Cannot connect to host localhost:11434` | The local Ollama server isn't running. Start it with `ollama serve` (or use the Ollama Cloud embeddings class instead). |
| Answer says it couldn't find the answer | The information genuinely isn't in the (retrieved part of the) transcript — this is the grounding behavior working as designed, not a bug. |
| Answers seem generic or off-topic | Try rephrasing the question; long videos with very similar chunks can dilute retrieval. Increase `k` in `get_retriever` for more context. |

---

## ⚠️ Limitations & Notes

- **English/Hindi only** — the transcript loader requests `["en", "hi"]`; videos without captions in these languages can't be indexed.
- **Embedding model coupling** — the store and retriever must use the *same* embedding model. If you change `embeddinggemma-2:270m` in one place, change it in the other (and re-index existing stores).
- **Separate indexes per video** — the web app indexes one video at a time per session and doesn't cross-search multiple videos; each video ID gets its own database directory.
- **Quality depends on captions** — auto-generated transcripts can contain mis-heard words, which both the index and the answers inherit.
- **`transcript.txt`** is just a sample transcript saved for reference — not modified by the app.
- **ChromaDB cache artifacts** (`chroma.db/` contents, `*.sqlite3`) and `.env` are gitignored; don't commit them.

---

## 🧭 Possible Improvements

- Swap local embeddings for `OllamaCloudEmbeddings` to remove the local server dependency.
- Cross-video querying: one shared collection with a `video_id` metadata filter.
- Timestamp-aware answers (publish the relevant segment of the video).
- Streaming responses in the chat UI.
- A conversation-aware retriever (rewrite follow-up questions before retrieval).

---

## 📄 License

No license has been added yet. Until one is chosen, all rights remain with the author.
