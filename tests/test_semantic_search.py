import pytest

from src.embeddings import HashingEmbeddingProvider
from src.semantic_search import (
    build_chroma_where,
    normalize_query_results,
    normalize_search_filters,
    similarity_from_distance,
    validate_embedding_provider_matches_collection,
)


class FakeCollection:
    def __init__(self, metadata: dict | None) -> None:
        self.metadata = metadata


def test_similarity_from_cosine_distance() -> None:
    assert similarity_from_distance(0.25) == 0.75


def test_normalize_query_results_returns_post_ids_scores_and_metadata() -> None:
    raw = {
        "ids": [["1", "2"]],
        "documents": [["China tariffs", "Oil policy"]],
        "metadatas": [[{"primary_topic": "tariff_trade"}, {"primary_topic": "oil_energy"}]],
        "distances": [[0.1, 0.3]],
    }

    result = normalize_query_results(raw)

    assert result[0]["post_id"] == "1"
    assert result[0]["score"] == 0.9
    assert result[0]["cleaned_text"] == "China tariffs"
    assert result[1]["metadata"]["primary_topic"] == "oil_energy"


def test_search_filters_ignore_all_and_build_chroma_where() -> None:
    filters = normalize_search_filters(
        {
            "primary_topic": "war_defense",
            "tone": "all",
            "market_relevance": "",
            "policy_direction": "escalation",
        }
    )

    assert filters == {"primary_topic": "war_defense", "policy_direction": "escalation"}
    assert build_chroma_where(filters) == {
        "$and": [{"primary_topic": "war_defense"}, {"policy_direction": "escalation"}]
    }


def test_search_filters_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported search filter"):
        normalize_search_filters({"unknown": "value"})


def test_provider_mismatch_raises_clear_error() -> None:
    provider = HashingEmbeddingProvider()
    collection = FakeCollection(metadata={"embedding_provider": "openai"})

    with pytest.raises(ValueError, match="Embedding provider mismatch"):
        validate_embedding_provider_matches_collection(provider, collection)
