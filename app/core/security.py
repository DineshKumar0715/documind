from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> bool:
    """
    Validate incoming API key if configured.
    In development mode or if no API key is enforced, requests are permitted.
    """
    if not settings.SECRET_KEY or settings.APP_ENV == "development":
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    if api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key provided",
        )
    return True
