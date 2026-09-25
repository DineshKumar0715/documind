import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.utils.logging import logger

class IngestionService:
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text and page metadata from PDF file."""
        pages_content = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_content.append({
                        "text": text.strip(),
                        "page_number": page_idx + 1
                    })
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
        return pages_content

    def extract_text_from_plain_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text or markdown files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [{"text": text.strip(), "page_number": 1}] if text.strip() else []
        except Exception as e:
            logger.error(f"Error reading plain file {file_path}: {e}")
            raise

    def process_file(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        file_type: str
    ) -> List[Dict[str, Any]]:
        """
        Parse file, split content into chunks, and return structured chunk metadata.
        """
        logger.info(f"Processing file: {filename} (type: {file_type}, doc_id: {document_id})")
        ext = Path(filename).suffix.lower()
        
        if ext == ".pdf" or file_type == "application/pdf":
            pages = self.extract_text_from_pdf(file_path)
        else:
            pages = self.extract_text_from_plain_file(file_path)

        if not pages:
            logger.warning(f"No text extracted from document {filename}")
            return []

        all_chunks: List[Dict[str, Any]] = []
        chunk_idx = 0

        for page in pages:
            raw_text = page["text"]
            page_num = page["page_number"]
            split_texts = self.splitter.split_text(raw_text)

            for txt in split_texts:
                if not txt.strip():
                    continue
                chunk_data = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk_idx,
                    "content": txt,
                    "char_count": len(txt),
                    "page_number": page_num,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_index": chunk_idx
                    }
                }
                all_chunks.append(chunk_data)
                chunk_idx += 1

        logger.info(f"Extracted {len(all_chunks)} chunks for document {filename}")
        return all_chunks
