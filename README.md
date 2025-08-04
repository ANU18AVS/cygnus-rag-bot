 Cygnus RAG Bot

A local Q&A bot built using LangChain, Ollama, ChromaDB, and Streamlit.

##  Features

- Upload multiple PDF files
- Embed documents into a vector DB using Ollama embeddings
- Ask natural language questions
- Retrieve source document chunks for answers

##  How to Run

1. Install dependencies:

```bash
pip install streamlit langchain chromadb
```

2. Start Ollama and run a supported model like `mistral`:

```bash
ollama run mistral
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open in browser at `http://localhost:8501`

## 📁 Folder Structure

```
cygnus_rag_bot/
│
├── app.py            # Main Streamlit application
├── README.md         # Project instructions
├── chroma_store/     # Stores Chroma vector DB
└── sample_docs/      # Upload your PDFs here
```




