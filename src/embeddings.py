"""LLM and embedding factories for Ollama and OpenAI."""

from __future__ import annotations

from typing import Any

from src.config import Settings, require_api_key
from src.utils import AppError


def build_embeddings(
    settings: Settings | None = None,
) -> Any:
    """
    Build the configured embedding model.

    Ollama:
        nomic-embed-text:latest

    OpenAI:
        text-embedding-3-small by default
    """

    settings = require_api_key(settings)

    try:

        if settings.provider == "ollama":

            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )

        from langchain_openai import OpenAIEmbeddings

        kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
        }

        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return OpenAIEmbeddings(
            **kwargs
        )

    except Exception as exc:

        raise AppError(
            "Could not initialize the embedding model. "
            f"Provider: {settings.provider}. "
            f"Model: {settings.embedding_model}. "
            f"Error: {exc}"
        ) from exc


def build_llm(
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> Any:
    """
    Build the configured chat language model.

    Ollama:
        llama3.2:latest

    OpenAI:
        gpt-4o-mini by default.
    """

    settings = require_api_key(settings)

    try:

        if settings.provider == "ollama":

            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )

        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.llm_model,
            "api_key": settings.openai_api_key,
            "temperature": temperature,
        }

        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return ChatOpenAI(
            **kwargs
        )

    except Exception as exc:

        raise AppError(
            "Could not initialize the language model. "
            f"Provider: {settings.provider}. "
            f"Model: {settings.llm_model}. "
            f"Error: {exc}"
        ) from exc