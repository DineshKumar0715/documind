import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from app.core.config import settings
from app.services.embeddings import get_embedding_function
from app.schemas.chat import Citation
from app.utils.logging import logger

COLLECTION_NAME = "documind_knowledge"

class RetrievalService:
    def __init__(self):
        self.embedding_fn = get_embedding_function()
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        
        # Initialize persistent chroma client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize LangChain Chroma vectorstore wrapper
        self.vector_store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        logger.info(f"Retrieval service initialized with ChromaDB at: {self.persist_dir}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add text chunks with metadata to ChromaDB."""
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{c['document_id']}_{c['chunk_index']}" for c in chunks]

        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} chunks to vector store collection '{COLLECTION_NAME}'")
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a given document from the vector store."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted vector chunks for document {document_id}")
            return True
        except Exception as e:
            logger.warning(f"Error deleting chunks for document {document_id}: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Citation]:
        """
        Retrieve most relevant context chunks for a given query with optional filtering.
        """
        k = top_k or settings.TOP_K_RESULTS
        min_score = score_threshold if score_threshold is not None else 0.0

        filter_dict = None
        if document_ids and len(document_ids) == 1:
            filter_dict = {"document_id": document_ids[0]}
        elif document_ids and len(document_ids) > 1:
            filter_dict = {"document_id": {"$in": document_ids}}

        try:
            # similarity_search_with_score returns (Document, distance)
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}")
            docs = self.vector_store.similarity_search(query=query, k=k, filter=filter_dict)
            results = [(d, 0.2) for d in docs]

        citations: List[Citation] = []
        for doc, dist in results:
            # Convert distance to normalized similarity score [0.0, 1.0]
            # Chroma default L2/cosine distance: closer means dist is smaller
            if dist is None:
                sim_score = 0.85
            else:
                sim_score = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, float(dist)))))

            if min_score > 0.0 and sim_score < min_score:
                continue

            meta = doc.metadata or {}
            citation = Citation(
                document_id=meta.get("document_id", "unknown"),
                document_name=meta.get("filename", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                content=doc.page_content,
                page_number=meta.get("page_number"),
                similarity_score=round(sim_score, 4)
            )
            citations.append(citation)

        logger.info(f"Retrieved {len(citations)} citations for query: '{query[:50]}...'")
        return citations
