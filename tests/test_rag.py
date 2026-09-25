import pytest
from app.db.database import SessionLocal, init_db
from app.services.rag import RAGService
from app.schemas.chat import ChatRequest

@pytest.mark.asyncio
async def test_rag_query_execution():
    init_db()
    db = SessionLocal()
    try:
        rag = RAGService(db)
        request = ChatRequest(
            message="Explain Retrieval-Augmented Generation (RAG) architecture."
        )
        response = await rag.answer_query(request)
        
        assert response is not None
        assert response.session_id is not None
        assert len(response.message) > 0
        assert response.latency_seconds >= 0.0
    finally:
        db.close()
