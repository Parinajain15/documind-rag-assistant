import pytest
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


def test_empty_document_raises_value_error():
    with pytest.raises(ValueError, match="No readable text"):
        DocumentRAGIndex([])


def test_unrelated_query_returns_empty_for_long_corpus():
    sample_chunks = [
        {"page": i, "text": f"Telecom network routing protocol specification section {i}."}
        for i in range(1, 10)
    ]
    index = DocumentRAGIndex(sample_chunks)
    results = index.retrieve("chocolate cake baking recipe", top_k=3)
    assert results == []