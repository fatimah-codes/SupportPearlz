"""
Configuration module for SupportPearlz.
Loads environment variables, validates settings with Pydantic, and provides central access.
"""

import os
import sys
from typing import Optional, Literal
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env from repository root if it exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application configuration and tunable parameters."""

    model_config = SettingsConfigDict(
        env_file=str(env_path),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider & Model Choices
    LLM_PROVIDER: Literal["google_genai", "openai", "mock"] = Field(
        default="google_genai",
        description="LLM Provider backend"
    )
    LLM_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Instruction-tuned Chat Model identifier"
    )
    EMBEDDING_PROVIDER: Literal["google_genai", "openai", "chroma_default", "mock"] = Field(
        default="chroma_default",
        description="Embedding model provider"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-004",
        description="Embedding model identifier"
    )
    TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Generation temperature (low and explicit)"
    )

    # API Keys
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")

    # Storage & Paths
    KNOWLEDGE_BASE_DIR: Path = Field(
        default=BASE_DIR / "data" / "knowledge_base",
        description="Source documents folder"
    )
    VECTOR_STORE_DIR: Path = Field(
        default=BASE_DIR / "data" / "vector_store",
        description="Persisted vector database directory"
    )
    COLLECTION_NAME: str = Field(
        default="pearlz_kb",
        description="Vector store collection name"
    )

    # Ingestion & Chunking parameters
    CHUNK_SIZE: int = Field(default=800, ge=100, le=4000, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=120, ge=0, le=1000, description="Overlap in characters")

    # Retrieval parameters
    RETRIEVAL_K: int = Field(default=4, ge=1, le=20, description="Number of chunks retrieved")
    # Relevance gate threshold (calibrated for local subword embeddings; set to 0.65+ for OpenAI/Gemini)
    SCORE_THRESHOLD: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Cosine similarity relevance gate threshold"
    )
    SEARCH_STRATEGY: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="similarity",
        description="Retriever search type"
    )

    # Context & Session Bounds
    MAX_CONTEXT_CHARS: int = Field(
        default=6000,
        description="Maximum characters of retrieved context injected into prompt"
    )
    MEMORY_WINDOW_TURNS: int = Field(
        default=5,
        description="Number of conversation turns retained in memory"
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Console and file log level")
    LOG_FILE_PATH: Path = Field(
        default=BASE_DIR / "supportpearlz.log",
        description="Log output path"
    )

    def validate_api_keys(self) -> None:
        """Validate presence of API key when not in mock mode."""
        if self.LLM_PROVIDER == "google_genai" and not (self.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")):
            raise ValueError(
                "Missing required environment variable: GOOGLE_API_KEY is required when LLM_PROVIDER='google_genai'. "
                "Add GOOGLE_API_KEY to your .env file or export it."
            )
        if self.LLM_PROVIDER == "openai" and not (self.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")):
            raise ValueError(
                "Missing required environment variable: OPENAI_API_KEY is required when LLM_PROVIDER='openai'. "
                "Add OPENAI_API_KEY to your .env file or export it."
            )

    def mask_secret(self, secret: Optional[str]) -> str:
        """Mask secrets for display in logs or smoke checks."""
        if not secret:
            return "[NOT SET]"
        if len(secret) <= 8:
            return "[MASKED]"
        return f"{secret[:4]}...{secret[-4:]} [MASKED]"

    def display_masked_summary(self) -> str:
        """Return formatted configuration with secrets safely masked."""
        lines = [
            "=" * 60,
            "SUPPORTPEARLZ CONFIGURATION STATUS",
            "=" * 60,
            f"  LLM Provider:       {self.LLM_PROVIDER}",
            f"  LLM Model:          {self.LLM_MODEL}",
            f"  Embedding Provider: {self.EMBEDDING_PROVIDER}",
            f"  Embedding Model:    {self.EMBEDDING_MODEL}",
            f"  Temperature:        {self.TEMPERATURE}",
            f"  Chunk Size/Overlap: {self.CHUNK_SIZE} / {self.CHUNK_OVERLAP}",
            f"  Retrieval k:        {self.RETRIEVAL_K}",
            f"  Score Threshold:    {self.SCORE_THRESHOLD}",
            f"  Search Strategy:    {self.SEARCH_STRATEGY}",
            f"  Max Context Chars:  {self.MAX_CONTEXT_CHARS}",
            f"  Vector Store Path:  {self.VECTOR_STORE_DIR}",
            f"  Collection Name:    {self.COLLECTION_NAME}",
            f"  Google API Key:     {self.mask_secret(self.GOOGLE_API_KEY or os.environ.get('GOOGLE_API_KEY'))}",
            f"  OpenAI API Key:     {self.mask_secret(self.OPENAI_API_KEY or os.environ.get('OPENAI_API_KEY'))}",
            f"  Log Level:          {self.LOG_LEVEL}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Singleton settings instance
settings = Settings()

if __name__ == "__main__":
    print(settings.display_masked_summary())
