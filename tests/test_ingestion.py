import tempfile
from pathlib import Path
import pytest
from app.services.ingestion import IngestionService

def test_plain_text_chunking():
    service = IngestionService(chunk_size=100, chunk_overlap=20)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("DocuMind is an AI-powered document intelligence and RAG platform. " * 10)
        temp_path = f.name

    try:
        chunks = service.process_file(
            document_id="test_doc_1",
            filename="sample.txt",
            file_path=temp_path,
            file_type="text/plain"
        )
        assert len(chunks) > 1
        assert chunks[0]["document_id"] == "test_doc_1"
        assert chunks[0]["chunk_index"] == 0
        assert "content" in chunks[0]
        assert chunks[0]["page_number"] == 1
    finally:
        Path(temp_path).unlink(missing_ok=True)

def test_empty_file_handling():
    service = IngestionService()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        chunks = service.process_file(
            document_id="test_doc_empty",
            filename="empty.txt",
            file_path=temp_path,
            file_type="text/plain"
        )
        assert chunks == []
    finally:
        Path(temp_path).unlink(missing_ok=True)
