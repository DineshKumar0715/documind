# DocuMind Enterprise Overview

DocuMind is an enterprise-grade AI-powered Document Intelligence and Retrieval-Augmented Generation (RAG) platform.

## Key Capabilities
1. Multi-Format Ingestion: Supports PDFs, Markdown files, Plain Text, JSON, and CSV documents.
2. Semantic Chunking: Splits documents recursively preserving headers, paragraph boundaries, and page numbers.
3. High-Performance Retrieval: Uses ChromaDB vector search with cosine/L2 distance scoring and top-k filtering.
4. Google Gemini LLM Integration: Uses Google Gemini models (Gemini 1.5 Flash, Gemini 1.5 Pro, and Gemini 2.0) with text-embedding-004 for fast, grounded, and fact-checked responses with explicit page citations.
5. Conversational Memory: Maintains multi-turn context across chat sessions in an embedded SQLite database.
