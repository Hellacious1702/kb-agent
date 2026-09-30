"""Document processing utilities for PDF, MD, TXT, DOCX files."""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

import fitz  # pymupdf
from docx import Document as DocxDocument
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """Extracted document content with metadata."""
    text: str
    pages: list[dict]  # List of {page_number, text, char_start, char_end}
    metadata: dict
    file_hash: str


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    def extract(self, file_path: Path) -> DocumentContent:
        """Extract text and metadata from a document."""
        pass

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        pass


class PDFProcessor(DocumentProcessor):
    """Process PDF documents using PyMuPDF."""

    def supports(self, content_type: str) -> bool:
        return content_type == "application/pdf"

    def extract(self, file_path: Path) -> DocumentContent:
        """Extract text from PDF with page-level metadata."""
        doc = fitz.open(file_path)
        pages = []
        full_text_parts = []
        char_offset = 0

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            page_char_start = char_offset
            char_offset += len(text)
            page_char_end = char_offset

            pages.append({
                "page_number": page_num,
                "text": text,
                "char_start": page_char_start,
                "char_end": page_char_end,
            })
            full_text_parts.append(text)

        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(file_path)

        # Extract metadata
        metadata = doc.metadata or {}
        metadata.update({
            "page_count": len(doc),
            "file_size": file_path.stat().st_size,
        })

        doc.close()

        return DocumentContent(
            text="\n\n".join(full_text_parts),
            pages=pages,
            metadata=metadata,
            file_hash=file_hash,
        )

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class DocxProcessor(DocumentProcessor):
    """Process DOCX documents using python-docx."""

    def supports(self, content_type: str) -> bool:
        return content_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]

    def extract(self, file_path: Path) -> DocumentContent:
        """Extract text from DOCX."""
        doc = DocxDocument(file_path)
        full_text_parts = []
        pages = []
        char_offset = 0

        # DOCX doesn't have native pages, treat as single page
        for para in doc.paragraphs:
            text = para.text
            if text.strip():
                full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)
        char_offset = len(full_text)

        pages.append({
            "page_number": 1,
            "text": full_text,
            "char_start": 0,
            "char_end": char_offset,
        })

        file_hash = self._calculate_hash(file_path)

        metadata = {
            "paragraph_count": len(doc.paragraphs),
            "file_size": file_path.stat().st_size,
        }

        return DocumentContent(
            text=full_text,
            pages=pages,
            metadata=metadata,
            file_hash=file_hash,
        )

    def _calculate_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class TextProcessor(DocumentProcessor):
    """Process plain text and markdown files."""

    def supports(self, content_type: str) -> bool:
        return content_type in ["text/plain", "text/markdown", "text/x-markdown"]

    def extract(self, file_path: Path) -> DocumentContent:
        """Extract text from plain text/markdown file."""
        text = file_path.read_text(encoding="utf-8")

        pages = [{
            "page_number": 1,
            "text": text,
            "char_start": 0,
            "char_end": len(text),
        }]

        file_hash = self._calculate_hash(file_path)

        metadata = {
            "file_size": file_path.stat().st_size,
            "line_count": text.count("\n") + 1,
        }

        return DocumentContent(
            text=text,
            pages=pages,
            metadata=metadata,
            file_hash=file_hash,
        )

    def _calculate_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class UnstructuredProcessor(DocumentProcessor):
    """Fallback processor using unstructured library for various formats."""

    def supports(self, content_type: str) -> bool:
        # Support many formats as fallback
        supported = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
            "text/markdown",
            "text/html",
            "application/rtf",
            "application/vnd.oasis.opendocument.text",
        ]
        return content_type in supported

    def extract(self, file_path: Path) -> DocumentContent:
        """Extract using unstructured partition."""
        elements = partition(filename=str(file_path))

        full_text_parts = []
        pages = []
        char_offset = 0

        # Group by page number if available
        page_elements = {}
        for el in elements:
            page_num = getattr(el.metadata, "page_number", 1) or 1
            if page_num not in page_elements:
                page_elements[page_num] = []
            page_elements[page_num].append(str(el))

        for page_num in sorted(page_elements.keys()):
            page_text = "\n\n".join(page_elements[page_num])
            page_char_start = char_offset
            char_offset += len(page_text)
            page_char_end = char_offset

            pages.append({
                "page_number": page_num,
                "text": page_text,
                "char_start": page_char_start,
                "char_end": page_char_end,
            })
            full_text_parts.append(page_text)

        file_hash = self._calculate_hash(file_path)

        metadata = {
            "element_count": len(elements),
            "file_size": file_path.stat().st_size,
        }

        return DocumentContent(
            text="\n\n".join(full_text_parts),
            pages=pages,
            metadata=metadata,
            file_hash=file_hash,
        )

    def _calculate_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class DocumentProcessorRegistry:
    """Registry for document processors."""

    def __init__(self):
        self._processors: list[DocumentProcessor] = [
            PDFProcessor(),
            DocxProcessor(),
            TextProcessor(),
            UnstructuredProcessor(),  # Fallback
        ]

    def get_processor(self, content_type: str) -> Optional[DocumentProcessor]:
        """Get the first processor that supports the content type."""
        for processor in self._processors:
            if processor.supports(content_type):
                return processor
        return None

    def extract(self, file_path: Path, content_type: str) -> DocumentContent:
        """Extract content using appropriate processor."""
        processor = self.get_processor(content_type)
        if not processor:
            raise ValueError(f"No processor found for content type: {content_type}")
        logger.info(f"Using {processor.__class__.__name__} for {content_type}")
        return processor.extract(file_path)


# Global registry instance
document_processor_registry = DocumentProcessorRegistry()