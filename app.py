"""
Simple RAG chatbot (terminal app).

Flow: Document Upload -> Parsing -> Chunking -> Embeddings -> Vector Storage -> Retrieval -> LLM Response

Usage:
    python app.py path/to/your/document.pdf
    python app.py path/to/your/document.txt
"""

import os
import sys
from getpass import getpass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import google.generativeai as genai


def get_embedding_model_name():
    """Different API keys/projects support different embedding models, so ask
    Gemini which one is actually available instead of hardcoding a name."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            return m.name
    raise RuntimeError("No embedding model available for this API key.")


def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


# --- 1. Document Upload + Parsing ---

def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


# --- 2. Chunking ---

def chunk_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py <path-to-document.pdf-or-.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")

    print(f"Loading '{file_path}' ...")
    docs = load_document(file_path)
    print(f"Loaded {len(docs)} page(s)/section(s).")

    chunks = chunk_document(docs)
    print(f"Split into {len(chunks)} chunks.")

    # --- 3. Embeddings + 4. Vector Storage ---
    embedding_model_name = get_embedding_model_name()
    print(f"Using embedding model: {embedding_model_name}")
    print("Embedding chunks and building the vector store (this may take a few seconds)...")
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    # --- 5. Retrieval ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- 6. LLM Response ---
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

    RAG_PROMPT = """Answer the question using ONLY the context below.
If the answer is not contained in the context, respond exactly with:
"I don't have that information in the document."
Do not use outside knowledge and do not guess.

Context:
{context}

Question: {question}

Answer:"""

    conversation_history = []

    print("\nReady! Ask questions about the document (type 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        system_message = SystemMessage(
            content=RAG_PROMPT.format(context=context, question=question)
        )
        messages = [system_message] + conversation_history + [HumanMessage(content=question)]

        response = llm.invoke(messages)
        answer = get_text(response)

        conversation_history.append(HumanMessage(content=question))
        conversation_history.append(AIMessage(content=answer))

        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
