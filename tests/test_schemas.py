import pytest
from pydantic import ValidationError

from src.schemas import NarrativeClassification, fallback_classification


def test_classification_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NarrativeClassification.model_validate(
            {
                "primary_topic": "tariff_trade",
                "tone": "aggressive",
                "entities": ["China"],
                "market_relevance": "high",
                "policy_direction": "escalation",
                "reason": "Tariff language is market relevant.",
                "ticker": "QQQ",
            }
        )


def test_fallback_classification_matches_required_defaults() -> None:
    fallback = fallback_classification()

    assert fallback.primary_topic.value == "other"
    assert fallback.tone.value == "neutral"
    assert fallback.entities == []
    assert fallback.market_relevance.value == "low"
    assert fallback.policy_direction.value == "neutral"


def test_entities_are_deduplicated_trimmed_and_capped() -> None:
    raw_entities = [" China ", "China", "x" * 100, *[f"Entity {i}" for i in range(20)]]

    result = NarrativeClassification.model_validate(
        {
            "primary_topic": "tariff_trade",
            "tone": "neutral",
            "entities": raw_entities,
            "market_relevance": "high",
            "policy_direction": "escalation",
            "reason": "Tariff language is market relevant.",
        }
    )

    assert result.entities[0] == "China"
    assert result.entities.count("China") == 1
    assert len(result.entities[1]) == 80
    assert len(result.entities) == 12
