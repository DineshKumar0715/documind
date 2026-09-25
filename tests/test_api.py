import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "DocuMind"

def test_dashboard_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "DocuMind" in response.text

def test_document_lifecycle():
    # 1. Upload a text document
    file_content = b"DocuMind provides enterprise grade document QA using LLMs and vector retrieval."
    files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
    
    upload_res = client.post("/api/v1/documents/upload", files=files)
    assert upload_res.status_code == 201
    doc_data = upload_res.json()["document"]
    doc_id = doc_data["id"]
    assert doc_data["filename"] == "test_doc.txt"
    assert doc_data["status"] == "indexed"
    assert doc_data["chunk_count"] >= 1

    # 2. List documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 3. Query chat RAG
    chat_res = client.post("/api/v1/chat", json={"message": "What does DocuMind provide?"})
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "session_id" in chat_data
    assert len(chat_data["message"]) > 0

    # 4. Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
