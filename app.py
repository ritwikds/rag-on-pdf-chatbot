import gradio as gr
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_groq import ChatGroq
import os
import uuid

# === Models ===
print("Groq key exists:", bool(os.getenv("GROQ_API_KEY")))
print("Groq key value:", os.getenv("GROQ_API_KEY"))

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# === Prompt Templates ===
rewrite_prompt = PromptTemplate.from_template("""
Rewrite the following question to improve document search relevance.

Original Question: {question}

Rewritten Search Query:""")
rewrite_chain = LLMChain(llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0), prompt=rewrite_prompt)

answer_prompt = PromptTemplate.from_template("""
Use the following context to answer the question. 
If you're unsure, say you don't know. Don't make things up.

Context:
{context}

Question:
{question}

Answer:""")
answer_chain = StuffDocumentsChain(
    llm_chain=LLMChain(llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0), prompt=answer_prompt),
    document_variable_name="context"
)

rerank_prompt = PromptTemplate.from_template("""
Given the following query and document, rate how relevant the document is to the query on a scale of 1-10.

Query:
{query}

Document:
{document}

Score (1-10):""")
rerank_chain = LLMChain(llm=ChatGroq(model_name="llama-3.1-8b-instant", temperature=0), prompt=rerank_prompt)

# === Global Vars ===
retriever = None
vector_store = None
memory = None

def create_vector_db_for_pdf(docs, file_id=None):
    # Create a unique collection per PDF
    collection_name = file_id or str(uuid.uuid4())

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=f"./chroma_dbs/{collection_name}"
    )
    vector_db.add_documents(docs)
    vector_db.persist()

    return vector_db

def load_pdf(file):
    global retriever, vector_store, memory

    yield "⏳ Processing PDF...", gr.update(visible=False)

    loader = PyMuPDFLoader(file.name)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)

    # Use filename or UUID as unique ID
    file_id = os.path.splitext(os.path.basename(file.name))[0]

    # Create per-document vector DB
    vector_store = create_vector_db_for_pdf(docs, file_id=file_id)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    yield "✅ PDF processed. Ask your question!", gr.update(visible=True)

def keyword_match(docs, query):
    keywords = set(re.findall(r'\b\w+\b', query.lower()))
    scored = []
    for doc in docs:
        content = doc.page_content.lower()
        matches = sum(1 for kw in keywords if kw in content)
        if matches > 0:
            scored.append((matches, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:4]]

def rerank_documents(query, docs):
    scored_docs = []
    for doc in docs:
        score_str = rerank_chain.run({"query": query, "document": doc.page_content})
        try:
            score = int(re.search(r"\d+", score_str).group())
        except:
            score = 5  # fallback
        scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:4]]

def respond(message, history):
    if not retriever:
        return "", history, history

    # 🔁 Adaptive query rewrite
    rewritten_query = rewrite_chain.run({"question": message})
    print(f"📝 Rewritten Query: {rewritten_query}")

    # 📏 Adaptive k
    k = 3 if len(rewritten_query.split()) <= 6 else 8
    retriever.search_kwargs["k"] = k
    sim_docs = retriever.get_relevant_documents(rewritten_query)

    # 🔍 Keyword matching
    all_docs = vector_store.similarity_search(rewritten_query, k=50)
    keyword_docs = keyword_match(all_docs, rewritten_query)

    # 🧩 Hybrid + Rerank
    combined_docs = list({doc.page_content: doc for doc in (sim_docs + keyword_docs)}.values())
    top_docs = rerank_documents(rewritten_query, combined_docs)

    # 📄 Print context for debugging
    context_texts = [doc.page_content for doc in top_docs]
    combined_context = "\n\n---\n\n".join(context_texts)

    # 🧠 Answer generation
    answer = answer_chain.run({"input_documents": top_docs, "question": message})

    history.append((f"**Q:** {message}\n\n**Context:**\n{combined_context}", f"**A:** {answer}"))
    return "", history, history

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Chat with your PDF ")

    # Upload section
    file_input = gr.File(label="Upload PDF")
    upload_status = gr.Markdown()
    chat_column = gr.Column(visible=False)

    with chat_column:
        chatbot = gr.Chatbot(label="Document QA Bot")
        msg = gr.Textbox(placeholder="Ask a question and press Enter", show_label=False)
        clear = gr.Button("Clear")

    # Connect file upload to PDF processing
    file_input.change(fn=load_pdf, inputs=file_input, outputs=[upload_status, chat_column])

    # --- Respond wrapper ---
    def respond_wrapper(message, history):
        # This wrapper avoids repeated outputs and clarifies structure
        updated_msg, updated_chat, _ = respond(message, history)
        return updated_msg, updated_chat

    # Submit question → respond → update chatbot and clear message
    msg.submit(fn=respond_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Clear button → reset everything
    def clear_all():
        return [], "", ""

    clear.click(fn=clear_all, outputs=[chatbot, msg, upload_status])

if __name__ == "__main__":
    demo.launch()
