import pandas as pd
import pytest

from src.build_chroma import (
    build_chroma_collection,
    chroma_rows,
    existing_collection_ids,
    get_chroma_client,
    get_collection,
    metadata_value,
    row_metadata,
    upsert_dataframe,
)
from src.embeddings import HashingEmbeddingProvider
from src.semantic_search import search_similar_posts


class FakeCollection:
    def __init__(self) -> None:
        self.upserts = []
        self._ids = set()

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)
        self._ids.update(kwargs["ids"])

    def count(self) -> int:
        return len(self._ids)

    def get(self, limit: int, offset: int, include: list[str]) -> dict:
        ids = sorted(self._ids)
        return {"ids": ids[offset : offset + limit]}


def test_chroma_rows_skips_empty_text_and_deduplicates_post_ids() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["1", "2", "2"],
            "cleaned_text": ["hello", "", "latest"],
        }
    )

    result = chroma_rows(df)

    assert result["post_id"].tolist() == ["1", "2"]
    assert result["cleaned_text"].tolist() == ["hello", "latest"]


def test_metadata_value_serializes_timestamps_and_missing_values() -> None:
    assert metadata_value(pd.Timestamp("2026-01-01T12:00:00Z")) == "2026-01-01T12:00:00+00:00"
    assert metadata_value(None) == ""


def test_row_metadata_keeps_required_chroma_fields() -> None:
    row = pd.Series(
        {
            "date": pd.Timestamp("2026-01-01"),
            "datetime": pd.Timestamp("2026-01-01T12:00:00Z"),
            "primary_topic": "tariff_trade",
            "tone": "threatening",
            "entities": "China,Tariffs",
            "market_relevance": "high",
            "policy_direction": "escalation",
            "is_president": True,
        }
    )

    metadata = row_metadata(row)

    assert metadata["primary_topic"] == "tariff_trade"
    assert metadata["is_president"] is True
    assert metadata["entities"] == "China,Tariffs"
    assert metadata["policy_direction"] == "escalation"


def test_upsert_dataframe_writes_embeddings_documents_and_metadata() -> None:
    collection = FakeCollection()
    provider = HashingEmbeddingProvider(dimensions=8)
    df = pd.DataFrame(
        {
            "post_id": ["1"],
            "cleaned_text": ["China tariffs"],
            "primary_topic": ["tariff_trade"],
            "policy_direction": ["escalation"],
        }
    )

    written = upsert_dataframe(collection, df, provider, batch_size=10)

    assert written == 1
    assert collection.upserts[0]["ids"] == ["1"]
    assert collection.upserts[0]["documents"] == ["China tariffs"]
    assert collection.upserts[0]["metadatas"][0]["primary_topic"] == "tariff_trade"
    assert len(collection.upserts[0]["embeddings"][0]) == 8


def test_upsert_dataframe_can_skip_existing_ids() -> None:
    collection = FakeCollection()
    collection._ids.add("1")
    provider = HashingEmbeddingProvider(dimensions=8)
    df = pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "cleaned_text": ["already here", "new text"],
        }
    )

    written = upsert_dataframe(collection, df, provider, skip_existing=True)

    assert written == 1
    assert collection.upserts[0]["ids"] == ["2"]
    assert existing_collection_ids(collection) == {"1", "2"}


def test_chroma_end_to_end_with_hashing_provider(tmp_path) -> None:
    pytest.importorskip("chromadb")
    input_path = tmp_path / "classified.parquet"
    df = pd.DataFrame(
        {
            "post_id": ["tariff-post", "oil-post"],
            "cleaned_text": ["china tariff", "oil energy"],
            "primary_topic": ["tariff_trade", "oil_energy"],
            "tone": ["neutral", "neutral"],
            "entities": ["China", "Oil"],
            "market_relevance": ["high", "medium"],
            "policy_direction": ["escalation", "neutral"],
            "is_president": [True, True],
        }
    )
    df.to_parquet(input_path, index=False)
    provider = HashingEmbeddingProvider(dimensions=64)

    written = build_chroma_collection(
        input_path=input_path,
        persist_dir=tmp_path / "chroma",
        collection_name="test_posts",
        embedding_provider=provider,
        batch_size=1,
        show_progress=False,
    )
    results = search_similar_posts(
        "china tariff",
        top_k=1,
        persist_dir=tmp_path / "chroma",
        collection_name="test_posts",
        embedding_provider=provider,
    )
    collection = get_collection(get_chroma_client(tmp_path / "chroma"), "test_posts")

    assert written == 2
    assert collection.count() == 2
    assert results[0]["post_id"] == "tariff-post"
