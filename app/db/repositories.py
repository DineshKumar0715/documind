import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.document import Document, DocumentChunk
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import Citation
from app.utils.logging import logger

def utc_now():
    return datetime.now(timezone.utc)

class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        file_size_bytes: int,
        status: str = "pending"
    ) -> Document:
        doc = Document(
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            status=status,
            created_at=utc_now(),
            updated_at=utc_now()
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_by_id(self, document_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> Tuple[List[Document], int]:
        query = self.db.query(Document).order_by(desc(Document.created_at))
        total = query.count()
        documents = query.offset(skip).limit(limit).all()
        return documents, total

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        doc = self.get_by_id(document_id)
        if not doc:
            return None
        doc.status = status
        doc.updated_at = utc_now()
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message is not None:
            doc.error_message = error_message
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def add_chunks(self, document_id: str, chunks_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        created_chunks = []
        now = utc_now()
        for c_data in chunks_data:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=c_data.get("chunk_index", 0),
                content=c_data.get("content", ""),
                char_count=len(c_data.get("content", "")),
                page_number=c_data.get("page_number"),
                metadata_json=json.dumps(c_data.get("metadata", {})),
                created_at=now
            )
            self.db.add(chunk)
            created_chunks.append(chunk)
        self.db.commit()
        return created_chunks

    def delete(self, document_id: str) -> bool:
        doc = self.get_by_id(document_id)
        if not doc:
            return False
        self.db.delete(doc)
        self.db.commit()
        return True


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, title: str = "New Conversation") -> ChatSession:
        now = utc_now()
        session = ChatSession(title=title, created_at=now, updated_at=now)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def list_sessions(self, skip: int = 0, limit: int = 50) -> Tuple[List[ChatSession], int]:
        query = self.db.query(ChatSession).order_by(desc(ChatSession.updated_at))
        total = query.count()
        sessions = query.offset(skip).limit(limit).all()
        return sessions, total

    def delete_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        self.db.delete(session)
        self.db.commit()
        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Citation]] = None,
        latency_seconds: float = 0.0
    ) -> ChatMessage:
        now = utc_now()
        citations_json = None
        if citations:
            citations_json = json.dumps([c.model_dump() for c in citations])
        
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            citations_json=citations_json,
            latency_seconds=latency_seconds,
            created_at=now
        )
        self.db.add(msg)
        
        # Also update session updated_at
        session = self.get_session(session_id)
        if session:
            session.updated_at = now
            if session.title == "New Conversation" and role == "user":
                session.title = content[:60] + ("..." if len(content) > 60 else "")
                
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
            .all()
        )
