"""Text chunking strategies for RAG."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_index: int
    page_number: int | None
    char_start: int
    char_end: int
    token_count: int | None = None
    metadata: dict | None = None


class Chunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, pages: list[dict]) -> list[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(Chunker):
    """
    Fixed-size chunking with overlap.
    
    WHY: Simple, predictable chunk sizes that work well for embedding models
    with fixed context windows. Overlap ensures context continuity across
    chunk boundaries. Good baseline strategy before trying semantic chunking.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, pages: list[dict]) -> list[Chunk]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        chunk_index = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence ending within the last 100 chars
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end

            chunk_text = text[start:end].strip()
            if not chunk_text:
                start = end
                continue

            # Find which page this chunk belongs to
            page_number = self._find_page_number(pages, start, end)
            char_start = start
            char_end = end

            chunks.append(Chunk(
                content=chunk_text,
                chunk_index=chunk_index,
                page_number=page_number,
                char_start=char_start,
                char_end=char_end,
                token_count=self._estimate_tokens(chunk_text),
            ))

            chunk_index += 1
            # Only apply overlap if we haven't reached the end
            if end >= text_len:
                break
            start = end - self.chunk_overlap
            start = max(start, 0)

        return chunks

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the last sentence boundary within [start, end]."""
        # Look for sentence endings: . ! ? followed by space or newline
        pattern = r'[.!?]\s+'
        matches = list(re.finditer(pattern, text[start:end]))
        if matches:
            # Return position after the last match
            last_match = matches[-1]
            return start + last_match.end()
        return end

    def _find_page_number(self, pages: list[dict], char_start: int, char_end: int) -> int | None:
        """Find the page number for a character range."""
        # Find page that contains the middle of the chunk
        mid = (char_start + char_end) // 2
        for page in pages:
            if page["char_start"] <= mid < page["char_end"]:
                return page["page_number"]
        # Fallback: find closest page
        for page in pages:
            if page["char_start"] <= char_start < page["char_end"]:
                return page["page_number"]
        return pages[0]["page_number"] if pages else None

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)."""
        return len(text) // 4


class SemanticChunker(Chunker):
    """
    Semantic chunking using sentence embeddings.
    
    WHY: Groups semantically related content together, preserving topic
    coherence within chunks. Better for retrieval quality because each
    chunk represents a complete thought/topic. More expensive but higher
    quality for complex documents.
    
    Strategy: Use sentence-transformers to embed sentences, then cluster
    adjacent sentences with high cosine similarity.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self._model = None

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to fixed-size")
                return None
        return self._model

    def chunk(self, text: str, pages: list[dict]) -> list[Chunk]:
        """Create semantic chunks based on sentence similarity."""
        model = self._get_model()
        if model is None:
            # Fallback to fixed-size
            fixed_chunker = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
            return fixed_chunker.chunk(text, pages)

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # Not enough sentences for semantic chunking
            fixed_chunker = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
            return fixed_chunker.chunk(text, pages)

        # Embed sentences
        sentence_texts = [s["text"] for s in sentences]
        embeddings = model.encode(sentence_texts)

        # Group sentences by similarity
        groups = self._group_by_similarity(sentences, embeddings)

        # Convert groups to chunks
        chunks = []
        for chunk_index, group in enumerate(groups):
            group_text = " ".join([s["text"] for s in group])
            char_start = group[0]["char_start"]
            char_end = group[-1]["char_end"]
            page_number = self._find_page_number(pages, char_start, char_end)

            chunks.append(Chunk(
                content=group_text,
                chunk_index=chunk_index,
                page_number=page_number,
                char_start=char_start,
                char_end=char_end,
                token_count=self._estimate_tokens(group_text),
            ))

        return chunks

    def _split_sentences(self, text: str) -> list[dict]:
        """Split text into sentences with character positions."""
        # Simple sentence splitting - can be improved with spaCy
        pattern = r'(?<=[.!?])\s+'
        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            sentence_text = text[last_end:match.end()].strip()
            if sentence_text:
                sentences.append({
                    "text": sentence_text,
                    "char_start": last_end,
                    "char_end": match.end(),
                })
            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append({
                    "text": remaining,
                    "char_start": last_end,
                    "char_end": len(text),
                })

        return sentences

    def _group_by_similarity(
        self,
        sentences: list[dict],
        embeddings: list[list[float]],
    ) -> list[list[dict]]:
        """Group adjacent sentences by cosine similarity."""
        import numpy as np

        groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            # Calculate cosine similarity with previous sentence
            prev_emb = np.array(embeddings[i - 1])
            curr_emb = np.array(embeddings[i])

            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)
            )

            # Check if we should start a new chunk
            current_text = " ".join([s["text"] for s in current_group])
            should_split = (
                similarity < self.similarity_threshold
                or len(current_text) >= self.chunk_size
            )

            if should_split and len(current_text) >= self.min_chunk_size:
                groups.append(current_group)
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        if current_group:
            groups.append(current_group)

        return groups

    def _find_page_number(self, pages: list[dict], char_start: int, char_end: int) -> int | None:
        mid = (char_start + char_end) // 2
        for page in pages:
            if page["char_start"] <= mid < page["char_end"]:
                return page["page_number"]
        for page in pages:
            if page["char_start"] <= char_start < page["char_end"]:
                return page["page_number"]
        return pages[0]["page_number"] if pages else None

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4


class HybridChunker(Chunker):
    """
    Hybrid chunking: semantic first, then fixed-size for long chunks.
    
    WHY: Best of both worlds - semantic coherence for normal content,
    but prevents extremely large chunks that exceed embedding model limits.
    Falls back to fixed-size splitting for chunks that are too large.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
    ):
        self.semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
        )
        self.fixed_chunker = FixedSizeChunker(chunk_size, chunk_overlap)

    def chunk(self, text: str, pages: list[dict]) -> list[Chunk]:
        """Apply semantic chunking, then split oversized chunks."""
        semantic_chunks = self.semantic_chunker.chunk(text, pages)

        final_chunks = []
        chunk_index = 0

        for chunk in semantic_chunks:
            if chunk.token_count and chunk.token_count > self.semantic_chunker.chunk_size // 4 * 1.5:
                # Chunk too large, split with fixed-size
                sub_chunks = self.fixed_chunker.chunk(chunk.content, pages)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    final_chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                chunk.chunk_index = chunk_index
                final_chunks.append(chunk)
                chunk_index += 1

        return final_chunks
