# 🧠 Research Assistant RAG with LangChain, Groq, FAISS

This project builds a simple Retrieval-Augmented Generation (RAG) system that:
- Reads a PDF file
- Splits it into chunks
- Stores embeddings using FAISS
- Uses Groq's LLM (LLaMA3) to answer questions about the document

## 🔧 Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Groq API key in a `.env` file:

```
GROQ_API_KEY=your-groq-api-key
```

3. Add your PDF document as `sample_paper.pdf` in the project directory.

4. Run the assistant:

```bash
python main.py
```

## 📦 Technologies Used
- LangChain
- Groq LLMs (`llama3-8b`)
- HuggingFace Embeddings
- FAISS for vector storage
- PyMuPDF + Unstructured for PDF loading
