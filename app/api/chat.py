"""API routes for chat/RAG queries."""

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_workspace
from app.core.database import get_db_session
from app.models import Query, Workspace
from app.services.rag_generator import rag_generator
from app.services.retriever import hybrid_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int | None = Field(default=10, ge=1, le=50)
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    citations: list[dict]
    sources: list[dict]
    token_usage: dict
    latency_ms: int


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Query the knowledge base."""
    import time
    start_time = time.time()

    # Retrieve relevant chunks
    sources = await hybrid_retriever.retrieve(
        query=request.question,
        workspace_id=workspace.id,
        top_k=request.top_k,
    )

    if not sources:
        return ChatResponse(
            answer="I don't have enough information to answer this question. No relevant documents found.",
            citations=[],
            sources=[],
            token_usage={},
            latency_ms=int((time.time() - start_time) * 1000),
        )

    # Generate answer
    response = await rag_generator.generate(
        question=request.question,
        sources=sources,
        stream=False,
    )

    # Save query for analytics
    query_record = Query(
        workspace_id=workspace.id,
        question=request.question,
        answer=response.answer,
        sources=response.citations,
        latency_ms=response.latency_ms,
        token_usage=response.token_usage,
    )
    session.add(query_record)
    await session.commit()

    return ChatResponse(
        answer=response.answer,
        citations=response.citations,
        sources=[
            {
                "chunk_id": s.chunk_id,
                "document_id": s.document_id,
                "page_number": s.page_number,
                "score": s.score,
                "source_type": s.source_type,
            }
            for s in response.sources
        ],
        token_usage=response.token_usage,
        latency_ms=response.latency_ms,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
):
    """Stream chat response."""
    # Retrieve relevant chunks
    sources = await hybrid_retriever.retrieve(
        query=request.question,
        workspace_id=workspace.id,
        top_k=request.top_k,
    )

    if not sources:
        async def empty_generator():
            yield 'data: {"type": "token", "content": "I don\'t have enough information to answer this question."}\n\n'
            yield 'data: {"type": "done", "response": {"answer": "I don\'t have enough information to answer this question.", "citations": [], "sources": [], "token_usage": {}, "latency_ms": 0}}\n\n'
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    async def generate_stream():
        full_answer = ""
        citations = []
        token_usage = {}
        latency_ms = 0

        async for event in rag_generator.generate_stream(
            question=request.question,
            sources=sources,
        ):
            if event["type"] == "token":
                full_answer += event["content"]
                yield f'data: {{"type": "token", "content": {event["content"]!r}}}\n\n'
            elif event["type"] == "done":
                citations = event["response"].citations
                token_usage = event["response"].token_usage
                latency_ms = event["response"].latency_ms

                # Save query
                query_record = Query(
                    workspace_id=workspace.id,
                    question=request.question,
                    answer=full_answer,
                    sources=citations,
                    latency_ms=latency_ms,
                    token_usage=token_usage,
                )
                session.add(query_record)
                await session.commit()

                import json
                yield f'data: {json.dumps({"type": "done", "response": {"answer": full_answer, "citations": citations, "sources": [], "token_usage": token_usage, "latency_ms": latency_ms}})}\n\n'
            elif event["type"] == "error":
                yield f'data: {{"type": "error", "error": {event["error"]!r}}}\n\n'

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.get("/history")
async def get_chat_history(
    workspace: Workspace = Depends(get_current_workspace),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Get chat history for workspace."""
    result = await session.execute(
        select(Query)
        .where(Query.workspace_id == workspace.id)
        .order_by(Query.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    queries = result.scalars().all()

    return [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "sources": q.sources,
            "latency_ms": q.latency_ms,
            "token_usage": q.token_usage,
            "created_at": q.created_at.isoformat(),
        }
        for q in queries
    ]
