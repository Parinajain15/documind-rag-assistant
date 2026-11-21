import os
import re
from typing import List, Dict
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def extract_chunks_from_pdf(pdf_file, chunk_size: int = 450, overlap: int = 100) -> List[Dict]:
    """Reads a PDF file object and splits text into overlapping chunks with page metadata."""
    reader = PdfReader(pdf_file)
    chunks = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean_text = re.sub(r"\s+", " ", raw_text).strip()
        if not clean_text:
            continue

        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            chunk_text = clean_text[start:end].strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "page": page_idx + 1,
                    "text": chunk_text
                })
            start += (chunk_size - overlap)

    return chunks


class DocumentRAGIndex:
    """Hybrid word + character n-gram TF-IDF index with full-context fallback for short docs."""

    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("No readable text could be extracted from this PDF.")
        self.chunks = chunks
        self.word_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        corpus = [c["text"] for c in chunks]
        self.word_matrix = self.word_vec.fit_transform(corpus)
        self.char_matrix = self.char_vec.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        # If the entire PDF is short (<= 6 chunks), return all chunks so nothing is missed
        if len(self.chunks) <= top_k:
            return [
                {"page": c["page"], "text": c["text"], "score": 1.0}
                for c in self.chunks
            ]

        w_scores = cosine_similarity(self.word_vec.transform([query]), self.word_matrix).flatten()
        c_scores = cosine_similarity(self.char_vec.transform([query]), self.char_matrix).flatten()
        combined = 0.65 * w_scores + 0.35 * c_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if combined[idx] > 0.09:
                results.append({
                    "page": self.chunks[idx]["page"],
                    "text": self.chunks[idx]["text"],
                    "score": round(float(combined[idx]), 4)
                })
        return results


def generate_grounded_answer(query: str, retrieved_chunks: List[Dict], api_key: str = None) -> str:
    """Generates an LLM answer strictly grounded in the retrieved PDF chunks with page citations."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing GROQ_API_KEY. Add it to your .env file or Advanced Settings.")

    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded PDF to answer this question."

    context_blocks = []
    for item in retrieved_chunks:
        context_blocks.append(f"[Page {item['page']}]: {item['text']}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a precise enterprise document assistant. Answer the user's question using ONLY "
        "the provided PDF excerpts below. Always cite page numbers like (Page X). "
        "If the answer is not explicitly supported by the excerpts, respond exactly with: "
        "'I cannot find this information in the uploaded document.' Do not guess or use outside knowledge."
    )

    user_prompt = f"PDF Excerpts:\n{context_text}\n\nQuestion: {query}"

    client = Groq(api_key=key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=500,
    )
    return chat_completion.choices[0].message.content