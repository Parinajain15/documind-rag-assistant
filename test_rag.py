from rag_engine import DocumentRAGIndex

def test_retrieval_and_citation_metadata():
    sample_chunks = [
        {"page": 1, "text": "Kalinga Institute of Industrial Technology offers B.Tech in CSE."},
        {"page": 2, "text": "Employees receive 12 days of annual paid sick leave."}
    ]
    index = DocumentRAGIndex(sample_chunks)
    results = index.retrieve("How many sick leaves are allowed?", top_k=1)
    assert len(results) == 1
    assert results[0]["page"] == 2
    assert "12 days" in results[0]["text"]