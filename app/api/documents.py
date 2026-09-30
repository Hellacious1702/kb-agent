"""API routes for document management."""

import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_workspace
from app.core.config import get_settings
from app.core.database import get_db_session
from app.models import Document, DocumentStatus, Workspace
from app.services.chunker import HybridChunker
from app.services.document_processor import document_processor_registry
from app.services.embedder import embedder
from app.services.retriever import hybrid_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["documents"])

# Upload directory
UPLOAD_DIR = Path("/tmp/kb-agent-uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed content types
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}

MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Upload and process a document."""
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}",
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    # Generate unique filename
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / workspace.id / unique_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    # Create document record
    document = Document(
        workspace_id=workspace.id,
        filename=unique_filename,
        original_filename=file.filename,
        content_type=file.content_type,
        file_size=len(content),
        file_path=str(file_path),
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    # Process document asynchronously (in background)
    # For now, process synchronously
    await process_document(document.id, session)

    return {
        "id": document.id,
        "filename": document.original_filename,
        "status": document.status.value,
    }


async def process_document(document_id: str, session: AsyncSession):
    """Process a document: extract, chunk, embed, store."""
    from app.models import Chunk

    # Get document
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        logger.error(f"Document {document_id} not found")
        return

    try:
        document.status = DocumentStatus.PROCESSING
        await session.commit()

        # Extract content
        file_path = Path(document.file_path)
        doc_content = document_processor_registry.extract(file_path, document.content_type)

        # Check for duplicate by hash
        existing = await session.execute(
            select(Document).where(
                Document.workspace_id == document.workspace_id,
                Document.meta.op("->>")("file_hash") == doc_content.file_hash,
                Document.id != document.id,
            )
        )
        if existing.scalar_one_or_none():
            document.status = DocumentStatus.FAILED
            document.error_message = "Duplicate document (same content hash)"
            await session.commit()
            return

        # Update document metadata
        document.meta = {**document.meta, "file_hash": doc_content.file_hash, **doc_content.metadata}
        await session.commit()

        # Chunk document
        chunker = HybridChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk(doc_content.text, doc_content.pages)

        # Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = await embedder.embed(chunk_texts)

        # Store chunks in database and vector store
        chunk_ids = []
        chunk_documents = []
        chunk_metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = str(uuid4())
            chunk_ids.append(chunk_id)

            chunk_record = Chunk(
                id=chunk_id,
                document_id=document.id,
                workspace_id=document.workspace_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_count=chunk.token_count,
                embedding=embedding,
                metadata={
                    "filename": document.original_filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                },
            )
            session.add(chunk_record)

            chunk_documents.append(chunk.content)
            chunk_metadatas.append({
                "workspace_id": document.workspace_id,
                "document_id": document.id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "filename": document.original_filename,
            })

        await session.commit()

        # Store in vector store and BM25
        await hybrid_retriever.add_chunks(
            workspace_id=document.workspace_id,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
        )

        document.status = DocumentStatus.COMPLETED
        document.processed_at = datetime.utcnow()
        await session.commit()

        logger.info(f"Processed document {document_id}: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        document.status = DocumentStatus.FAILED
        document.error_message = str(e)
        await session.commit()


@router.get("")
async def list_documents(
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List documents in workspace."""
    query = select(Document).where(Document.workspace_id == workspace.id)

    if status_filter:
        query = query.where(Document.status == DocumentStatus(status_filter))

    query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(query)
    documents = result.scalars().all()

    return [
        {
            "id": d.id,
            "filename": d.original_filename,
            "content_type": d.content_type,
            "file_size": d.file_size,
            "status": d.status.value,
            "error_message": d.error_message,
            "created_at": d.created_at.isoformat(),
            "processed_at": d.processed_at.isoformat() if d.processed_at else None,
        }
        for d in documents
    ]


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Get document details."""
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace.id,
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "filename": document.original_filename,
        "content_type": document.content_type,
        "file_size": document.file_size,
        "status": document.status.value,
        "error_message": document.error_message,
        "meta": document.meta,
        "created_at": document.created_at.isoformat(),
        "processed_at": document.processed_at.isoformat() if document.processed_at else None,
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a document and its chunks."""
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace.id,
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file
    file_path = Path(document.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete from vector store (handled by cascade in DB, but we need to remove from ChromaDB)
    # For now, we'll rebuild BM25 after deletion
    await session.delete(document)
    await session.commit()

    # Rebuild BM25 index for workspace
    await hybrid_retriever.rebuild_bm25_for_workspace(workspace.id)


