from __future__ import annotations

import hashlib
import math
import os
import re
import time
import warnings
from collections.abc import Iterable
from typing import Any, Protocol

from src.config import DEFAULT_EMBEDDING_MODEL


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
MAX_OPENAI_EMBEDDING_CHARS = 6000


class EmbeddingProvider(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalHashingEmbeddingProvider:
    name = "local_hashing"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: Any | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 1.0,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        self.client = client or self.create_client()
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", model)
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        safe_texts = truncate_embedding_texts(texts)
        response = self.create_embeddings_with_retry(safe_texts)
        return [item.embedding for item in response.data]

    def create_client(self) -> Any:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed.") from exc

        return OpenAI(api_key=api_key)

    def create_embeddings_with_retry(self, texts: list[str]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self.client.embeddings.create(model=self.model, input=texts, timeout=30)
            except Exception as exc:
                if not is_transient_openai_error(exc) or attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_base_seconds * (2**attempt))
        raise RuntimeError("OpenAI embedding request failed after retries.")


HashingEmbeddingProvider = LocalHashingEmbeddingProvider


def tokenize(text: str) -> Iterable[str]:
    for match in TOKEN_PATTERN.finditer(text.lower()):
        yield match.group(0)


def create_embedding_provider(kind: str = "auto") -> EmbeddingProvider:
    normalized = kind.strip().lower()
    if normalized == "hashing":
        return LocalHashingEmbeddingProvider()
    if normalized == "local_hashing":
        return LocalHashingEmbeddingProvider()
    if normalized == "openai":
        return OpenAIEmbeddingProvider()
    if normalized != "auto":
        raise ValueError("Embedding provider must be one of: auto, openai, hashing, local_hashing.")

    try:
        return OpenAIEmbeddingProvider()
    except RuntimeError:
        warnings.warn(
            "OpenAI embeddings are unavailable; falling back to local hashing embeddings. "
            "Use --embedding-provider openai to require OpenAI embeddings.",
            stacklevel=2,
        )
        return LocalHashingEmbeddingProvider()


def truncate_embedding_texts(texts: list[str]) -> list[str]:
    return [text[:MAX_OPENAI_EMBEDDING_CHARS] for text in texts]


def is_transient_openai_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIError",
        "APIStatusError",
    }:
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)
