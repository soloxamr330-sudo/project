"""Application configuration for Ollama and OpenAI providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "faiss"
CACHE_DIR = DATA_DIR / "cache"
REGISTRY_PATH = DATA_DIR / "registry.json"


load_dotenv(ROOT_DIR / ".env")


class ConfigError(RuntimeError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    provider: str

    openai_api_key: str

    llm_model: str
    embedding_model: str

    top_k: int

    openai_base_url: str | None

    ollama_base_url: str

    chunk_size: int
    chunk_overlap: int


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw)

    except ValueError as exc:
        raise ConfigError(
            f"{name} must be an integer."
        ) from exc

    if value <= 0:
        raise ConfigError(
            f"{name} must be greater than 0."
        )

    return value


def get_settings() -> Settings:
    provider = (
        os.getenv(
            "LLM_PROVIDER",
            "ollama",
        )
        .strip()
        .lower()
    )

    if provider not in {
        "ollama",
        "openai",
    }:
        raise ConfigError(
            "LLM_PROVIDER must be either 'ollama' or 'openai'."
        )

    openai_api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    openai_base_url = (
        os.getenv(
            "OPENAI_BASE_URL",
            "",
        ).strip()
        or None
    )

    ollama_base_url = (
        os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        )
        .strip()
        or "http://127.0.0.1:11434"
    )

    llm_model = (
        os.getenv(
            "LLM_MODEL",
            "llama3.2:latest",
        )
        .strip()
        or "llama3.2:latest"
    )

    embedding_model = (
        os.getenv(
            "EMBEDDING_MODEL",
            "nomic-embed-text:latest",
        )
        .strip()
        or "nomic-embed-text:latest"
    )

    return Settings(
        provider=provider,

        openai_api_key=openai_api_key,

        llm_model=llm_model,
        embedding_model=embedding_model,

        top_k=_int_env(
            "TOP_K",
            4,
        ),

        openai_base_url=openai_base_url,

        ollama_base_url=ollama_base_url,

        chunk_size=_int_env(
            "CHUNK_SIZE",
            1000,
        ),

        chunk_overlap=_int_env(
            "CHUNK_OVERLAP",
            200,
        ),
    )


def require_api_key(
    settings: Settings | None = None,
) -> Settings:
    """
    Validate provider configuration.

    Ollama does not require an API key.
    OpenAI requires OPENAI_API_KEY.
    """

    settings = settings or get_settings()

    if settings.provider == "ollama":
        return settings

    if not settings.openai_api_key:
        raise ConfigError(
            "OPENAI_API_KEY is missing. "
            "Set it in .env or switch LLM_PROVIDER to ollama."
        )

    return settings


def ensure_data_dirs() -> None:
    for path in (
        DATA_DIR,
        UPLOAD_DIR,
        VECTOR_DIR,
        CACHE_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )