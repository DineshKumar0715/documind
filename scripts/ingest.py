import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.db.database import SessionLocal, init_db
from app.db.repositories import DocumentRepository
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.utils.logging import logger

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}

def ingest_file(file_path: Path, db) -> bool:
    """Ingest a single file into SQLite and ChromaDB."""
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Skipping unsupported file: {file_path.name}")
        return False

    file_size = file_path.stat().st_size
    filename = file_path.name
    file_type = file_path.suffix.lower()

    doc_repo = DocumentRepository(db)
    
    # Check if existing document
    doc = doc_repo.create(
        filename=filename,
        file_path=str(file_path),
        file_type=file_type,
        file_size_bytes=file_size,
        status="indexing"
    )

    try:
        ingestion_service = IngestionService()
        chunks = ingestion_service.process_file(
            document_id=doc.id,
            filename=filename,
            file_path=str(file_path),
            file_type=file_type
        )

        if not chunks:
            doc_repo.update_status(doc.id, status="failed", error_message="No readable text found")
            logger.warning(f"No chunks extracted from {filename}")
            return False

        doc_repo.add_chunks(doc.id, chunks)

        retrieval_service = RetrievalService()
        retrieval_service.add_chunks(chunks)

        doc_repo.update_status(doc.id, status="indexed", chunk_count=len(chunks))
        logger.info(f" Successfully ingested '{filename}' -> {len(chunks)} chunks.")
        return True

    except Exception as e:
        logger.error(f"Failed to ingest {filename}: {e}")
        doc_repo.update_status(doc.id, status="failed", error_message=str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="DocuMind Batch Document Ingestion CLI")
    parser.add_argument("--path", "-p", type=str, default="./data/documents", help="Path to file or directory to ingest")
    parser.add_argument("--chunk-size", "-c", type=int, default=None, help="Override chunk size")
    parser.add_argument("--chunk-overlap", "-o", type=int, default=None, help="Override chunk overlap")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    target_path = Path(args.path)
    if not target_path.exists():
        logger.error(f"Path does not exist: {target_path}")
        sys.exit(1)

    logger.info(f"Starting batch ingestion for path: {target_path.resolve()}")
    
    files_to_process = []
    if target_path.is_file():
        files_to_process.append(target_path)
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files_to_process.extend(target_path.glob(f"*{ext}"))
            files_to_process.extend(target_path.glob(f"**/*{ext}"))
        files_to_process = list(set(files_to_process))

    if not files_to_process:
        logger.warning(f"No supported document files found in {target_path}")
        sys.exit(0)

    logger.info(f"Found {len(files_to_process)} document(s) to process.")
    success_count = 0

    for file_p in files_to_process:
        if ingest_file(file_p, db):
            success_count += 1

    db.close()
    logger.info(f"Ingestion complete: {success_count}/{len(files_to_process)} files processed successfully.")

if __name__ == "__main__":
    main()
