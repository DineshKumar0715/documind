import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.db.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="pending")  # pending, indexing, indexed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False, default=0)
    page_number = Column(Integer, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc_id={self.document_id} idx={self.chunk_index}>"
