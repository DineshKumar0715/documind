import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import ChatRepository
from app.services.rag import RAGService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatMessageResponse,
    Citation
)
from app.utils.logging import logger

router = APIRouter(prefix="/chat", tags=["Chat & RAG"])

@router.post("", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Execute a RAG query over indexed documents.
    Returns grounded AI answers with source citations, page numbers, and similarity metrics.
    """
    rag_service = RAGService(db)
    return await rag_service.answer_query(request)


@router.post("/stream")
async def chat_query_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Stream RAG response tokens in real-time using Server-Sent Events (SSE).
    """
    rag_service = RAGService(db)
    return StreamingResponse(
        rag_service.stream_query(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: Optional[ChatSessionCreate] = None,
    db: Session = Depends(get_db)
):
    """Create a new conversational chat session."""
    title = body.title if body and body.title else "New Conversation"
    chat_repo = ChatRepository(db)
    session = chat_repo.create_session(title=title)
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[]
    )


@router.get("/sessions", response_model=ChatSessionListResponse)
def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all chat sessions."""
    chat_repo = ChatRepository(db)
    sessions, total = chat_repo.list_sessions(skip=skip, limit=limit)
    return ChatSessionListResponse(
        total=total,
        sessions=[ChatSessionResponse.model_validate(s) for s in sessions]
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get chat session details and all historical messages with citations."""
    chat_repo = ChatRepository(db)
    session = chat_repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    parsed_messages: List[ChatMessageResponse] = []
    for msg in session.messages:
        citations_list: List[Citation] = []
        if msg.citations_json:
            try:
                c_dicts = json.loads(msg.citations_json)
                citations_list = [Citation(**c) for c in c_dicts]
            except Exception:
                pass

        parsed_messages.append(ChatMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            citations=citations_list,
            latency_seconds=msg.latency_seconds,
            created_at=msg.created_at
        ))

    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=parsed_messages
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session and all of its associated messages."""
    chat_repo = ChatRepository(db)
    deleted = chat_repo.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {"message": f"Session {session_id} deleted successfully."}
