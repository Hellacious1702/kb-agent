"""API routes for workspace and API key management."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_workspace
from app.core.database import get_db_session
from app.models import APIKey, Workspace, WorkspaceStatus
from app.utils.auth import workspace_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceCreate(BaseModel):
    """Workspace creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = None


class WorkspaceUpdate(BaseModel):
    """Workspace update request."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[WorkspaceStatus] = None


class WorkspaceResponse(BaseModel):
    """Workspace response model."""
    id: str
    name: str
    slug: str
    description: Optional[str]
    status: str
    settings: dict
    created_at: str
    updated_at: str


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    expires_days: Optional[int] = Field(None, ge=1, le=3650)


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: str
    name: str
    key_prefix: str
    full_key: Optional[str] = None  # Only returned on creation
    is_active: bool
    expires_at: Optional[str]
    created_at: str


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    request: WorkspaceCreate,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new workspace."""
    # Check slug uniqueness
    existing = await session.execute(
        select(Workspace).where(Workspace.slug == request.slug)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workspace slug already exists",
        )

    workspace = Workspace(
        name=request.name,
        slug=request.slug,
        description=request.description,
        status=WorkspaceStatus.ACTIVE,
    )
    session.add(workspace)
    await session.commit()
    await session.refresh(workspace)

    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        description=workspace.description,
        status=workspace.status.value,
        settings=workspace.settings,
        created_at=workspace.created_at.isoformat(),
        updated_at=workspace.updated_at.isoformat(),
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List all workspaces (admin)."""
    result = await session.execute(
        select(Workspace)
        .order_by(Workspace.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    workspaces = result.scalars().all()

    return [
        WorkspaceResponse(
            id=w.id,
            name=w.name,
            slug=w.slug,
            description=w.description,
            status=w.status.value,
            settings=w.settings,
            created_at=w.created_at.isoformat(),
            updated_at=w.updated_at.isoformat(),
        )
        for w in workspaces
    ]


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get workspace by ID."""
    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        description=workspace.description,
        status=workspace.status.value,
        settings=workspace.settings,
        created_at=workspace.created_at.isoformat(),
        updated_at=workspace.updated_at.isoformat(),
    )


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    request: WorkspaceUpdate,
    session: AsyncSession = Depends(get_db_session),
):
    """Update workspace."""
    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    if request.name is not None:
        workspace.name = request.name
    if request.description is not None:
        workspace.description = request.description
    if request.status is not None:
        workspace.status = request.status

    await session.commit()
    await session.refresh(workspace)

    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        slug=workspace.slug,
        description=workspace.description,
        status=workspace.status.value,
        settings=workspace.settings,
        created_at=workspace.created_at.isoformat(),
        updated_at=workspace.updated_at.isoformat(),
    )


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Delete workspace (cascades to documents, chunks, API keys)."""
    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    await session.delete(workspace)
    await session.commit()


# API Key routes
@router.post("/{workspace_id}/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    workspace_id: str,
    request: APIKeyCreate,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new API key for a workspace."""
    # Verify workspace exists
    result = await session.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Create API key
    from datetime import datetime, timedelta
    full_key, key_prefix, key_hash = workspace_auth.create_api_key(request.name)
    
    expires_at = None
    if request.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_days)

    api_key = APIKey(
        workspace_id=workspace_id,
        name=request.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        expires_at=expires_at,
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        full_key=full_key,  # Only returned once!
        is_active=api_key.is_active,
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        created_at=api_key.created_at.isoformat(),
    )


@router.get("/{workspace_id}/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    workspace_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """List API keys for a workspace (without full keys)."""
    result = await session.execute(
        select(APIKey)
        .where(APIKey.workspace_id == workspace_id)
        .order_by(APIKey.created_at.desc())
    )
    api_keys = result.scalars().all()

    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            full_key=None,  # Never return full key in list
            is_active=k.is_active,
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            created_at=k.created_at.isoformat(),
        )
        for k in api_keys
    ]


@router.delete("/{workspace_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    workspace_id: str,
    key_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Revoke an API key."""
    result = await session.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.workspace_id == workspace_id,
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await session.commit()