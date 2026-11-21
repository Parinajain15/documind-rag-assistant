import os
import streamlit as st
from rag_engine import extract_chunks_from_pdf, DocumentRAGIndex, generate_grounded_answer

st.set_page_config(page_title="DocuMind RAG Assistant", page_icon="📄", layout="wide")

st.title("📄 DocuMind — Grounded PDF Q&A Assistant")
st.caption("Upload any PDF and ask questions. Answers are strictly grounded in your document with page citations.")

# Sidebar for PDF upload & API status
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.header("2. System Status")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Server API Key active")
    else:
        st.error("❌ Server API Key missing in environment")

    if uploaded_file:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Reading PDF and building search index..."):
                try:
                    chunks = extract_chunks_from_pdf(uploaded_file)
                    st.session_state.rag_index = DocumentRAGIndex(chunks)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Question Form
with st.form(key="qa_form", clear_on_submit=True):
    prompt = st.text_input("Ask a question about your uploaded PDF:")
    submitted = st.form_submit_button("Ask Document")

if submitted and prompt.strip():
    if "rag_index" not in st.session_state:
        st.warning("Please upload a PDF in the left sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching document and generating grounded answer..."):
            try:
                retrieved = st.session_state.rag_index.retrieve(prompt, top_k=6)
                answer = generate_grounded_answer(prompt, retrieved)
            except Exception as e:
                answer = f"Error: {e}"
                retrieved = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved
        })

# Display Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.info(f"**🤖 DocuMind:**\n\n{msg['content']}")
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved PDF Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}** *(Similarity: {src['score']})*")
                    st.caption(src["text"])
        st.markdown("---")