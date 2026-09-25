import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Application Configuration
    APP_NAME: str = "DocuMind"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "documind_secret_key_default_2026"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_OUTPUT_TOKENS: int = 2048

    # Database & Storage
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'documind.db'}"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "documents")

    # Document Chunking & Retrieval Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 4
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    def ensure_directories(self) -> None:
        """Create necessary data directories if they do not exist."""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        db_path = Path(self.DATABASE_URL.replace("sqlite:///", ""))
        if not db_path.is_absolute():
            db_path = BASE_DIR / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
