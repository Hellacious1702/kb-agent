"""Tests for document processor."""

import tempfile
from pathlib import Path

from app.services.document_processor import (
    DocxProcessor,
    PDFProcessor,
    TextProcessor,
    document_processor_registry,
)


class TestTextProcessor:
    """Tests for TextProcessor."""

    def test_extract_txt(self):
        """Test extracting plain text."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world\nThis is a test document.\nWith multiple lines.")
            f.flush()
            file_path = Path(f.name)

        try:
            processor = TextProcessor()
            content = processor.extract(file_path)

            assert "Hello world" in content.text
            assert "test document" in content.text
            assert len(content.pages) == 1
            assert content.pages[0]["page_number"] == 1
            assert content.metadata["line_count"] == 3
        finally:
            file_path.unlink()

    def test_extract_md(self):
        """Test extracting markdown."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\n\nParagraph with **bold** text.")
            f.flush()
            file_path = Path(f.name)

        try:
            processor = TextProcessor()
            content = processor.extract(file_path)

            assert "Heading" in content.text
            assert "bold" in content.text
        finally:
            file_path.unlink()

    def test_supports(self):
        """Test content type support."""
        processor = TextProcessor()
        assert processor.supports("text/plain")
        assert processor.supports("text/markdown")
        assert processor.supports("text/x-markdown")
        assert not processor.supports("application/pdf")


class TestDocxProcessor:
    """Tests for DocxProcessor."""

    def test_supports(self):
        """Test content type support."""
        processor = DocxProcessor()
        assert processor.supports(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert processor.supports("application/msword")
        assert not processor.supports("text/plain")


class TestPDFProcessor:
    """Tests for PDFProcessor."""

    def test_supports(self):
        """Test content type support."""
        processor = PDFProcessor()
        assert processor.supports("application/pdf")
        assert not processor.supports("text/plain")


class TestDocumentProcessorRegistry:
    """Tests for DocumentProcessorRegistry."""

    def test_get_processor(self):
        """Test getting processor for content type."""
        registry = document_processor_registry

        pdf_processor = registry.get_processor("application/pdf")
        assert pdf_processor is not None
        assert isinstance(pdf_processor, PDFProcessor)

        docx_processor = registry.get_processor(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert docx_processor is not None
        assert isinstance(docx_processor, DocxProcessor)

        txt_processor = registry.get_processor("text/plain")
        assert txt_processor is not None
        assert isinstance(txt_processor, TextProcessor)

        unknown = registry.get_processor("application/unknown")
        assert unknown is None
