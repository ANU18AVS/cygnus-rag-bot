
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- Constants ---
CHROMA_DIR = "chroma_store"
DOCS_DIR = "sample_docs"
MODEL_NAME = "mistral"

# --- Ensure folders exist ---
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# --- Function: Load and embed documents ---
def embed_documents():
    st.info("🔄 Reading and embedding PDF documents...")
    all_docs = []

    # Load PDF files
    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
            all_docs.extend(loader.load())

    if not all_docs:
        st.warning("⚠️ No PDFs found in 'sample_docs'. Please upload documents first.")
        return

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Generate embeddings and store in Chroma
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()

    st.success(f"✅ Embedded {len(chunks)} chunks from {len(all_docs)} document(s).")

# --- Function: Create Retrieval-QA chain ---
def create_qa_chain():
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model=MODEL_NAME)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# --- Streamlit UI ---
st.set_page_config(page_title="Local RAG Bot", layout="centered")
st.title("🧠 Local AI Q&A Bot")
st.caption("Built with Ollama + LangChain + ChromaDB")

# --- Upload documents section ---
uploaded_files = st.file_uploader("📄 Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("📁 Files uploaded. Now click 'Embed Documents' to process them.")

# --- Embed button ---
if st.button("🔁 Embed Documents"):
    embed_documents()

# --- Chat section (only if DB exists) ---
if os.path.exists(os.path.join(CHROMA_DIR, "index")):
    qa_chain = create_qa_chain()

    user_question = st.text_input("💬 Ask a question based on the uploaded documents:")

    if user_question:
        with st.spinner("Generating response..."):
            result = qa_chain(user_question)
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown("### ✅ Answer:")
        st.write(answer)

        st.markdown("### 📚 Sources:")
        for doc in sources:
            page = doc.metadata.get("page", "?")
            src = doc.metadata.get("source", "Unknown")
            st.markdown(f"**Page {page}** — `{src}`")
            st.code(doc.page_content[:300] + "...")

else:
    st.info("ℹ️ No embedded data found. Upload and embed PDFs to enable the chat.")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Cygnus Internship | Task 1 – RAG Bot using LangChain + Ollama")
