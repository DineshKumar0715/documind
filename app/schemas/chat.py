from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

def utc_now():
    return datetime.now(timezone.utc)

class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    similarity_score: float

class ChatRequest(BaseModel):
    message: str = Field(..., description="The query or prompt from the user")
    session_id: Optional[str] = Field(default=None, description="Optional chat session ID for multi-turn history")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional filter to specific document IDs")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")
    stream: Optional[bool] = Field(default=False, description="Stream the response tokens")

class ChatResponse(BaseModel):
    session_id: str
    message: str
    citations: List[Citation] = []
    model: str
    latency_seconds: float
    created_at: datetime = Field(default_factory=utc_now)

class ChatStreamChunk(BaseModel):
    token: str
    done: bool = False
    session_id: Optional[str] = None
    citations: Optional[List[Citation]] = None
    latency_seconds: Optional[float] = None

class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Conversation"

class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    citations: List[Citation] = []
    latency_seconds: Optional[float] = 0.0
    created_at: datetime

class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []

class ChatSessionListResponse(BaseModel):
    total: int
    sessions: List[ChatSessionResponse]
