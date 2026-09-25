from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class DocumentChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int
    content: str
    char_count: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]

class DocumentDetailResponse(DocumentResponse):
    chunks: List[DocumentChunkSchema] = []

class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentResponse
