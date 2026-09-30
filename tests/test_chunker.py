"""Tests for chunking strategies."""

from app.services.chunker import FixedSizeChunker, HybridChunker, SemanticChunker


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 10  # ~240 chars
        pages = [{"page_number": 1, "char_start": 0, "char_end": len(text)}]
        
        chunks = chunker.chunk(text, pages)
        
        assert len(chunks) > 1
        assert all(c.chunk_index == i for i, c in enumerate(chunks))
        assert all(c.page_number == 1 for c in chunks)
        assert all(len(c.content) <= 120 for c in chunks)  # chunk_size + overlap buffer

    def test_overlap(self):
        """Test that chunks have overlap."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 120
        pages = [{"page_number": 1, "char_start": 0, "char_end": len(text)}]
        
        chunks = chunker.chunk(text, pages)
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap = chunks[i].char_end - chunks[i + 1].char_start
            assert overlap >= 8  # Allow some variance

    def test_sentence_boundary(self):
        """Test that chunks break at sentence boundaries."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        pages = [{"page_number": 1, "char_start": 0, "char_end": len(text)}]
        
        chunks = chunker.chunk(text, pages)
        
        # Each chunk should end at sentence boundary (approximately)
        for chunk in chunks[:-1]:  # Last chunk may not end at boundary
            assert chunk.content.rstrip().endswith((".", "!", "?"))


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_fallback_to_fixed_size(self):
        """Test fallback when sentence-transformers not available."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 10
        pages = [{"page_number": 1, "char_start": 0, "char_end": len(text)}]
        
        chunks = chunker.chunk(text, pages)
        
        # Should still produce chunks even without sentence-transformers
        assert len(chunks) > 0


class TestHybridChunker:
    """Tests for HybridChunker."""

    def test_hybrid_chunking(self):
        """Test hybrid chunking produces valid chunks."""
        chunker = HybridChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 10
        pages = [{"page_number": 1, "char_start": 0, "char_end": len(text)}]
        
        chunks = chunker.chunk(text, pages)
        
        assert len(chunks) > 0
        assert all(c.chunk_index == i for i, c in enumerate(chunks))