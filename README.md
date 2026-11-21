# 📄 DocuMind — Grounded PDF Q&A Assistant (RAG)

An interactive Retrieval-Augmented Generation (RAG) web application that enables users to upload PDF documents and ask natural-language questions with **page-level source citations** and **strict hallucination guardrails**.

🔗 **Live Demo:** [DocuMind on Streamlit Cloud](https://documind-rag-assistant-obojysfjkclfxjtxomzjqn.streamlit.app/)

---

## 🚀 Key Features

- **PDF Ingestion & Overlapping Chunking (`PyPDF2`):** Extracts text page-by-page and splits it into overlapping chunks (`450` chars, `100` overlap) while preserving exact page metadata.
- **Hybrid Lexical + Character N-Gram Retrieval (`scikit-learn`):** Combines word-level TF-IDF (`ngram_range=(1,2)`) and character n-gram TF-IDF (`char_wb`, `(3,5)`) with cosine similarity (`threshold > 0.09`) to handle acronyms, domain terms, and partial matches while filtering out unrelated queries.
- **Strict Hallucination Control (`Groq` LLM):** Enforces low-temperature (`0.1`) grounded prompting so answers rely exclusively on retrieved excerpts and explicitly refuse unsupported queries.
- **Transparent Source Citations & Session Controls (`Streamlit`):** Displays inline `(Page X)` citations alongside expandable source passages, similarity scores, and one-click chat reset.
- **Automated Unit Testing (`Pytest`):** Covers retrieval ranking, page metadata preservation, empty-document validation, and out-of-domain query filtering.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend UI:** Streamlit
- **Retrieval & Vector Search:** `scikit-learn` (Hybrid TF-IDF), `numpy`
- **LLM Inference:** Groq API (`openai/gpt-oss-120b`)
- **Document Parsing:** `PyPDF2`
- **Testing & Config:** `pytest`, `python-dotenv`

---

## 📂 Project Structure

```text
documind-rag-assistant/
├── app.py              # Streamlit web interface & session state management
├── rag_engine.py       # PDF chunking, hybrid TF-IDF index & grounded LLM generation
├── test_rag.py         # Pytest unit tests for retrieval, edge cases & page metadata
├── requirements.txt    # Production dependencies
├── .gitignore          # Excludes .env and virtual environment artifacts
└── README.md           # Project documentation