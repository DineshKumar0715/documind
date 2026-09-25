import pytest
from app.services.retrieval import RetrievalService

def test_retrieval_service_add_and_query():
    retrieval = RetrievalService()
    
    test_chunks = [
        {
            "document_id": "doc_retrieval_test",
            "filename": "quantum_computing.txt",
            "chunk_index": 0,
            "content": "Quantum computers use qubits which can exist in superposition states.",
            "char_count": 68,
            "page_number": 1,
            "metadata": {
                "document_id": "doc_retrieval_test",
                "filename": "quantum_computing.txt",
                "page_number": 1,
                "chunk_index": 0
            }
        },
        {
            "document_id": "doc_retrieval_test",
            "filename": "quantum_computing.txt",
            "chunk_index": 1,
            "content": "Superconducting circuits and trapped ions are leading qubit modalities.",
            "char_count": 72,
            "page_number": 1,
            "metadata": {
                "document_id": "doc_retrieval_test",
                "filename": "quantum_computing.txt",
                "page_number": 1,
                "chunk_index": 1
            }
        }
    ]

    added_count = retrieval.add_chunks(test_chunks)
    assert added_count == 2

    # Query
    results = retrieval.retrieve(
        query="What is superposition in quantum computing?",
        top_k=2,
        document_ids=["doc_retrieval_test"]
    )
    assert len(results) > 0
    assert any("qubit" in r.content.lower() or "quantum" in r.content.lower() for r in results)

    # Cleanup
    retrieval.delete_document_chunks("doc_retrieval_test")
