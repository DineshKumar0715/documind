import time
import json
from typing import Optional, List, AsyncGenerator
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repositories import ChatRepository
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    ChatStreamChunk
)
from app.utils.logging import logger

class RAGService:
    def __init__(self, db: Session):
        self.db = db
        self.chat_repo = ChatRepository(db)
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()

    async def answer_query(self, request: ChatRequest) -> ChatResponse:
        """Process a RAG query and return structured ChatResponse."""
        start_time = time.time()
        
        # 1. Ensure or create chat session
        session_id = request.session_id
        if not session_id:
            session = self.chat_repo.create_session(title=request.message[:50])
            session_id = session.id
        else:
            session = self.chat_repo.get_session(session_id)
            if not session:
                session = self.chat_repo.create_session(title=request.message[:50])
                session_id = session.id

        # 2. Record user message
        self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )

        # 3. Retrieve relevant context citations
        citations = self.retrieval_service.retrieve(
            query=request.message,
            top_k=request.top_k or settings.TOP_K_RESULTS,
            document_ids=request.document_ids,
            score_threshold=0.0
        )

        # 4. Fetch recent conversation history
        chat_history = self.chat_repo.get_recent_messages(session_id=session_id, limit=8)

        # 5. Generate grounded answer
        answer = await self.generation_service.generate_response(
            query=request.message,
            citations=citations,
            chat_history=chat_history
        )

        latency = round(time.time() - start_time, 3)

        # 6. Record assistant message with citations
        self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=citations,
            latency_seconds=latency
        )

        return ChatResponse(
            session_id=session_id,
            message=answer,
            citations=citations,
            model=settings.GEMINI_MODEL,
            latency_seconds=latency
        )

    async def stream_query(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream RAG response tokens via Server-Sent Events (SSE)."""
        start_time = time.time()
        
        # 1. Ensure session
        session_id = request.session_id
        if not session_id:
            session = self.chat_repo.create_session(title=request.message[:50])
            session_id = session.id
        else:
            session = self.chat_repo.get_session(session_id)
            if not session:
                session = self.chat_repo.create_session(title=request.message[:50])
                session_id = session.id

        # 2. Add user message
        self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )

        # 3. Retrieve context
        citations = self.retrieval_service.retrieve(
            query=request.message,
            top_k=request.top_k or settings.TOP_K_RESULTS,
            document_ids=request.document_ids,
            score_threshold=0.0
        )

        # 4. Get history
        chat_history = self.chat_repo.get_recent_messages(session_id=session_id, limit=8)

        # 5. Stream tokens
        full_response_parts = []
        async for token in self.generation_service.generate_stream(
            query=request.message,
            citations=citations,
            chat_history=chat_history
        ):
            full_response_parts.append(token)
            chunk_data = ChatStreamChunk(
                token=token,
                done=False,
                session_id=session_id
            )
            yield f"data: {chunk_data.model_dump_json()}\n\n"

        latency = round(time.time() - start_time, 3)
        complete_answer = "".join(full_response_parts)

        # 6. Save assistant message
        self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=complete_answer,
            citations=citations,
            latency_seconds=latency
        )

        # 7. Final done event with citations
        final_chunk = ChatStreamChunk(
            token="",
            done=True,
            session_id=session_id,
            citations=citations,
            latency_seconds=latency
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
