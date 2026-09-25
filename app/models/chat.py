import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.db.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} title={self.title}>"

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)  # JSON string of source citations
    latency_seconds = Column(Float, nullable=True, default=0.0)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role}>"
