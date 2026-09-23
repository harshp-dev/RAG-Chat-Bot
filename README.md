# Simple RAG Chatbot (POC)

A minimal end-to-end RAG pipeline:

```
Document Upload -> Parsing -> Chunking -> Embeddings -> Vector Storage -> Retrieval -> LLM Response
```

Same stack as the reference notebook: **LangChain + Google Gemini**. The only real addition
is that this loads an actual file (PDF or .txt) instead of a hardcoded string.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

You'll need a Gemini API key (same as the one used in the training notebook). It will prompt
you for it the first time you run the app — it is not stored anywhere.

## Run

```bash
python app.py sample_handbook.txt
```

or point it at your own PDF:

```bash
python app.py path\to\your\file.pdf
```

Then just ask questions in the terminal. Type `exit` to quit.

## How it maps to the RAG flow

| Step | Where in `app.py` |
|---|---|
| Document Upload | `sys.argv[1]` — path passed on the command line |
| Parsing | `load_document()` — `PyPDFLoader` for PDFs, `TextLoader` for .txt |
| Chunking | `chunk_document()` — `RecursiveCharacterTextSplitter` |
| Embeddings | `GoogleGenerativeAIEmbeddings` |
| Vector Storage | `InMemoryVectorStore` (in-memory, no server to set up) |
| Retrieval | `vector_store.as_retriever(...)` |
| LLM Response | `ChatGoogleGenerativeAI` with a "only answer from context" prompt |

Conversation history is also kept turn-to-turn, so you can ask a follow-up question and it
still remembers what you asked before (same idea as Section 8 of the training notebook).
