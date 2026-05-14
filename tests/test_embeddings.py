import pytest

from src.embeddings import (
    MAX_OPENAI_EMBEDDING_CHARS,
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


class TransientEmbeddingError(Exception):
    status_code = 429


class FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [FakeEmbeddingItem(embedding) for embedding in embeddings]


class FakeEmbeddingsClient:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0
        self.inputs = []

    def create(self, **kwargs):
        self.calls += 1
        self.inputs.append(kwargs["input"])
        if self.calls <= self.failures_before_success:
            raise TransientEmbeddingError("rate limited")
        return FakeEmbeddingResponse([[1.0, 0.0, 0.0] for _ in kwargs["input"]])


class FakeOpenAIClient:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.embeddings = FakeEmbeddingsClient(failures_before_success)


def test_hashing_embedding_is_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(dimensions=16)

    first = provider.embed_texts(["China tariffs"])[0]
    second = provider.embed_texts(["China tariffs"])[0]

    assert first == second
    assert len(first) == 16
    assert round(sum(value * value for value in first), 8) == 1.0


def test_hashing_embedding_empty_text_is_zero_vector() -> None:
    provider = HashingEmbeddingProvider(dimensions=8)

    assert provider.embed_texts([""])[0] == [0.0] * 8


def test_openai_embedding_provider_retries_transient_errors() -> None:
    client = FakeOpenAIClient(failures_before_success=1)
    provider = OpenAIEmbeddingProvider(client=client, retry_base_seconds=0)

    embeddings = provider.embed_texts(["China tariffs"])

    assert embeddings == [[1.0, 0.0, 0.0]]
    assert client.embeddings.calls == 2


def test_openai_embedding_provider_truncates_long_text() -> None:
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(client=client, retry_base_seconds=0)

    provider.embed_texts(["x" * (MAX_OPENAI_EMBEDDING_CHARS + 10)])

    assert len(client.embeddings.inputs[0][0]) == MAX_OPENAI_EMBEDDING_CHARS


def test_openai_embedding_provider_raises_non_transient_errors() -> None:
    class PermanentEmbeddingsClient(FakeEmbeddingsClient):
        def create(self, **kwargs):
            raise ValueError("bad request")

    client = FakeOpenAIClient()
    client.embeddings = PermanentEmbeddingsClient()
    provider = OpenAIEmbeddingProvider(client=client, retry_base_seconds=0)

    with pytest.raises(ValueError, match="bad request"):
        provider.embed_texts(["China tariffs"])
