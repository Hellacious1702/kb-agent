"""Additional API routes for admin functionality."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_workspace
from app.core.database import get_db_session
from app.models import Chunk, Document, DocumentStatus, Query as QueryModel, Workspace

router = APIRouter(prefix="/workspaces", tags=["admin"])


@router.get("/{workspace_id}/stats")
async def get_workspace_stats(
    workspace_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Get workspace statistics."""
    # Verify workspace matches
    if workspace.id != workspace_id:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Count documents
    doc_result = await session.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    document_count = doc_result.scalar() or 0

    # Count chunks
    chunk_result = await session.execute(
        select(func.count(Chunk.id)).where(Chunk.workspace_id == workspace_id)
    )
    chunk_count = chunk_result.scalar() or 0

    # Count API keys
    from app.models import APIKey
    api_key_result = await session.execute(
        select(func.count(APIKey.id)).where(APIKey.workspace_id == workspace_id)
    )
    api_key_count = api_key_result.scalar() or 0

    # Count queries in last 24h
    from datetime import datetime, timedelta
    day_ago = datetime.utcnow() - timedelta(hours=24)
    query_result = await session.execute(
        select(func.count(QueryModel.id)).where(
            QueryModel.workspace_id == workspace_id,
            QueryModel.created_at >= day_ago,
        )
    )
    query_count_24h = query_result.scalar() or 0

    return {
        "document_count": document_count,
        "chunk_count": chunk_count,
        "api_key_count": api_key_count,
        "query_count_24h": query_count_24h,
    }


@router.get("/{workspace_id}/chunks")
async def list_chunks(
    workspace_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List chunks for a workspace."""
    if workspace.id != workspace_id:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    result = await session.execute(
        select(Chunk, Document.original_filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.workspace_id == workspace_id)
        .order_by(Document.original_filename, Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    rows = result.all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_filename": filename,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "token_count": chunk.token_count,
            "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
        }
        for chunk, filename in rows
    ]


@router.post("/{workspace_id}/reindex")
async def reindex_workspace(
    workspace_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Re-index all documents in a workspace (rebuild BM25)."""
    if workspace.id != workspace_id:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Trigger BM25 rebuild via retriever
    from app.services.retriever import hybrid_retriever
    await hybrid_retriever.rebuild_bm25_for_workspace(workspace_id)

    return {"message": "Re-indexing completed"}


@router.get("/{workspace_id}/documents")
async def list_documents_admin(
    workspace_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List documents with chunk counts for admin."""
    if workspace.id != workspace_id:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Get documents with chunk counts
    from sqlalchemy import func
    result = await session.execute(
        select(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Document.id == Chunk.document_id)
        .where(Document.workspace_id == workspace_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = result.all()

    return [
        {
            "id": doc.id,
            "filename": doc.original_filename,
            "content_type": doc.content_type,
            "file_size": doc.file_size,
            "status": doc.status.value,
            "chunk_count": chunk_count,
            "created_at": doc.created_at.isoformat(),
        }
        for doc, chunk_count in rows
    ]