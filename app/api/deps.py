"""API route dependencies."""

from typing import Optional
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.models import Workspace
from app.utils.auth import workspace_auth


async def get_current_workspace(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_db_session),
) -> Workspace:
    """
    Get current workspace from API key or JWT token.
    
    Supports both:
    - Authorization: Bearer <jwt_token>
    - X-API-Key: sk_<prefix>_<random>
    """
    workspace = None

    if x_api_key:
        workspace = await workspace_auth.authenticate_api_key(session, x_api_key)
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        workspace = await workspace_auth.authenticate_token(session, token)

    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if workspace.status.value != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace is not active",
        )

    return workspace


async def get_current_workspace_optional(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    session: AsyncSession = Depends(get_db_session),
) -> Optional[Workspace]:
    """Get current workspace if authenticated, otherwise None."""
    try:
        return await get_current_workspace(authorization, x_api_key, session)
    except HTTPException:
        return None