"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/kb_agent",
        description="PostgreSQL connection URL",
    )

    # ChromaDB
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_collection_name: str = Field(
        default="kb_documents", description="ChromaDB collection name"
    )

    # LLM (via LiteLLM)
    litellm_api_key: str = Field(default="", description="LiteLLM API key")
    litellm_base_url: str = Field(
        default="https://api.openai.com/v1", description="LiteLLM base URL"
    )
    litellm_model: str = Field(default="gpt-4o-mini", description="LLM model for generation")
    litellm_embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )

    # Authentication
    secret_key: str = Field(
        default="change-me-in-production", description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )

    # App
    app_host: str = Field(default="0.0.0.0", description="App host")
    app_port: int = Field(default=8000, description="App port")
    debug: bool = Field(default=True, description="Debug mode")

    # Document Processing
    chunk_size: int = Field(default=1000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Chunk overlap in characters")
    max_file_size_mb: int = Field(default=50, description="Max file size in MB")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()