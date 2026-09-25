from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings
from app.utils.logging import logger

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a SQLAlchemy session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Initialize all database tables defined in models."""
    logger.info("Initializing database tables...")
    import app.models.document # noqa: F401
    import app.models.chat # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
