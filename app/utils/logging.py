import logging
import sys
from typing import Optional

def setup_logger(name: str = "documind", level: Optional[str] = None) -> logging.Logger:
    """Configure and return a structured logger."""
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

logger = setup_logger("documind")
