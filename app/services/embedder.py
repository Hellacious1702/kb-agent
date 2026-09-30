"""Embedding service using LiteLLM."""

import logging
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Configure litellm
litellm.api_key = settings.litellm_api_key
litellm.api_base = settings.litellm_base_url


class Embedder:
    """
    Embedding service using LiteLLM for provider-agnostic embeddings.
    
    WHY: LiteLLM provides a unified interface for multiple embedding providers
    (OpenAI, Azure, Cohere, HuggingFace, etc.). This allows switching providers
    without code changes. We use retry logic for resilience against transient failures.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.model = model or settings.litellm_embedding_model
        self.dimensions = dimensions

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []

        try:
            response = await litellm.aembedding(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
            
            embeddings = [item["embedding"] for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings with dim {len(embeddings[0]) if embeddings else 0}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []

    def get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        # Known dimensions for common models
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return self.dimensions or dimensions_map.get(self.model, 1536)


# Global embedder instance
embedder = Embedder()