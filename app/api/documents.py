import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import DocumentRepository
from app.core.config import settings
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentUploadResponse
)
from app.utils.logging import logger

router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and index a document (PDF, TXT, MD, JSON, CSV).
    Extracts text, creates semantic chunks, generates embeddings, and saves to vector store.
    """
    filename = file.filename or "unnamed_document"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Ensure upload directory
    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    saved_file_path = upload_path / filename
    # Avoid collisions if file exists
    if saved_file_path.exists():
        stem = Path(filename).stem
        saved_file_path = upload_path / f"{stem}_{os.urandom(4).hex()}{ext}"

    # Save to disk
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(saved_file_path)
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save file on server.")

    # Create document record in database
    doc_repo = DocumentRepository(db)
    doc = doc_repo.create(
        filename=filename,
        file_path=str(saved_file_path),
        file_type=file.content_type or ext,
        file_size_bytes=file_size,
        status="indexing"
    )

    try:
        # Process and chunk document
        ingestion_service = IngestionService()
        chunks = ingestion_service.process_file(
            document_id=doc.id,
            filename=filename,
            file_path=str(saved_file_path),
            file_type=file.content_type or ext
        )

        if not chunks:
            doc_repo.update_status(doc.id, status="failed", error_message="No readable text found in document.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract any text from the uploaded document."
            )

        # Store chunks in SQLite
        doc_repo.add_chunks(doc.id, chunks)

        # Store in ChromaDB vector store
        retrieval_service = RetrievalService()
        retrieval_service.add_chunks(chunks)

        # Update document status to indexed
        updated_doc = doc_repo.update_status(doc.id, status="indexed", chunk_count=len(chunks))

        return DocumentUploadResponse(
            message=f"Document '{filename}' successfully uploaded and indexed ({len(chunks)} chunks).",
            document=DocumentResponse.model_validate(updated_doc)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}", exc_info=True)
        doc_repo.update_status(doc.id, status="failed", error_message=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to index document: {str(e)}")


@router.get("", response_model=DocumentListResponse)
def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all indexed documents with pagination."""
    doc_repo = DocumentRepository(db)
    docs, total = doc_repo.list_all(skip=skip, limit=limit)
    return DocumentListResponse(
        total=total,
        documents=[DocumentResponse.model_validate(d) for d in docs]
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Retrieve details and chunk contents for a specific document."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentDetailResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document, its database records, vector embeddings, and physical file."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # 1. Delete vectors from ChromaDB
    retrieval_service = RetrievalService()
    retrieval_service.delete_document_chunks(document_id)

    # 2. Delete physical file if exists
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except Exception as e:
        logger.warning(f"Could not remove physical file {doc.file_path}: {e}")

    # 3. Delete from SQLite DB
    doc_repo.delete(document_id)

    return {"message": f"Document '{doc.filename}' (ID: {document_id}) was successfully deleted."}
