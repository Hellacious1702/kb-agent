"""RAG Generator with citation formatting."""

import logging
from dataclasses import dataclass

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.services.retriever import RetrievalResult

logger = logging.getLogger(__name__)
settings = get_settings()

# Configure litellm
litellm.api_key = settings.litellm_api_key
litellm.api_base = settings.litellm_base_url


@dataclass
class RAGResponse:
    """Response from RAG generation."""
    answer: str
    citations: list[dict]
    sources: list[RetrievalResult]
    token_usage: dict
    latency_ms: int


class CitationFormatter:
    """
    Format citations for RAG responses.
    
    WHY: Citations must be traceable to source documents with enough detail
    for users to verify. We include document name, chunk index, and page
    number. Format: [DocName, Chunk N, Page X] for inline citations.
    """

    @staticmethod
    def format_inline_citation(
        document_filename: str,
        chunk_index: int,
        page_number: int | None,
    ) -> str:
        """Format an inline citation marker."""
        parts = [document_filename, f"Chunk {chunk_index}"]
        if page_number is not None:
            parts.append(f"Page {page_number}")
        return f"[{', '.join(parts)}]"

    @staticmethod
    def format_source_list(sources: list[RetrievalResult]) -> list[dict]:
        """Format sources for display in source sidebar."""
        formatted = []
        for i, source in enumerate(sources):
            metadata = source.metadata
            formatted.append({
                "index": i + 1,
                "document_id": source.document_id,
                "document_filename": metadata.get("filename", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_number": source.page_number,
                "char_start": source.char_start,
                "char_end": source.char_end,
                "score": round(source.score, 4),
                "source_type": source.source_type,
                "preview": source.content[:300] + "..." if len(source.content) > 300 else source.content,
            })
        return formatted


class RAGGenerator:
    """
    RAG Generator with streaming support and citation formatting.
    
    WHY: Separate retrieval from generation. Use retrieved context to ground
    the LLM response. Include citations inline and provide full source
    details for the sidebar. Streaming improves perceived latency.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.model = model or settings.litellm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.citation_formatter = CitationFormatter()

    def _build_system_prompt(self) -> str:
        """Build system prompt for RAG."""
        return """You are a helpful assistant that answers questions based on the provided context.
        
Rules:
1. ONLY use information from the provided context to answer.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Cite your sources inline using the citation format: [DocumentName, Chunk N, Page X]
4. Be concise but thorough.
5. If multiple sources support a claim, cite all relevant sources."""

    def _build_context(self, sources: list[RetrievalResult]) -> str:
        """Build context string from retrieved sources."""
        context_parts = []
        for i, source in enumerate(sources):
            citation = self.citation_formatter.format_inline_citation(
                source.metadata.get("filename", f"Doc {source.document_id[:8]}"),
                source.metadata.get("chunk_index", 0),
                source.page_number,
            )
            context_parts.append(
                f"Source {i+1} {citation}:\n{source.content}\n"
            )
        return "\n".join(context_parts)

    def _build_messages(self, question: str, sources: list[RetrievalResult]) -> list[dict]:
        """Build messages for LLM."""
        context = self._build_context(sources)

        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        question: str,
        sources: list[RetrievalResult],
        stream: bool = False,
    ) -> RAGResponse:
        """
        Generate answer with citations.
        
        Args:
            question: User question
            sources: Retrieved source chunks
            stream: Whether to stream the response
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        import time
        start_time = time.time()

        messages = self._build_messages(question, sources)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
            )

            if stream:
                # For streaming, we'd need to handle async iteration
                # This is a simplified non-streaming version
                pass

            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Format citations for the answer
            formatted_sources = self.citation_formatter.format_source_list(sources)

            latency_ms = int((time.time() - start_time) * 1000)

            return RAGResponse(
                answer=answer,
                citations=formatted_sources,
                sources=sources,
                token_usage=token_usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(
        self,
        question: str,
        sources: list[RetrievalResult],
    ):
        """
        Generate streaming answer.
        
        Yields:
            Dict with either {"type": "token", "content": "..."} or
            {"type": "done", "response": RAGResponse}
        """
        import time
        start_time = time.time()

        messages = self._build_messages(question, sources)

        try:
            stream = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            full_answer = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield {"type": "token", "content": token}

            # Build final response
            formatted_sources = self.citation_formatter.format_source_list(sources)
            token_usage = {}  # Not available in streaming
            latency_ms = int((time.time() - start_time) * 1000)

            response = RAGResponse(
                answer=full_answer,
                citations=formatted_sources,
                sources=sources,
                token_usage=token_usage,
                latency_ms=latency_ms,
            )
            yield {"type": "done", "response": response}

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"type": "error", "error": str(e)}


# Global generator instance
rag_generator = RAGGenerator()
