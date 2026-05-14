import pandas as pd
import pytest

from src.classify import (
    classify_dataframe,
    classify_dataframe_incremental,
    classify_text,
    validate_classification,
)


def test_validate_classification_accepts_valid_json() -> None:
    result = validate_classification(
        """
        {
          "primary_topic": "tariff_trade",
          "tone": "threatening",
          "entities": ["China", "Tariffs"],
          "market_relevance": "high",
          "policy_direction": "escalation",
          "reason": "The post discusses tariff threats involving China."
        }
        """
    )

    assert result.primary_topic.value == "tariff_trade"


def test_validate_classification_accepts_fenced_json_with_prose() -> None:
    result = validate_classification(
        """
        Sure, here is the JSON:

        ```json
        {
          "primary_topic": "crypto",
          "tone": "neutral",
          "entities": [],
          "market_relevance": "low",
          "policy_direction": "neutral",
          "reason": "The post references crypto."
        }
        ```
        """
    )

    assert result.primary_topic.value == "crypto"


def test_validate_classification_extracts_first_balanced_json_object() -> None:
    result = validate_classification(
        """
        Classification:
        {
          "primary_topic": "rates_usd",
          "tone": "neutral",
          "entities": ["Fed"],
          "market_relevance": "medium",
          "policy_direction": "uncertainty",
          "reason": "The post references interest rates."
        }
        Done.
        """
    )

    assert result.primary_topic.value == "rates_usd"


def test_classify_text_retries_once_then_succeeds() -> None:
    responses = iter(
        [
            "not json",
            """
            {
              "primary_topic": "oil_energy",
              "tone": "neutral",
              "entities": ["Oil"],
              "market_relevance": "medium",
              "policy_direction": "uncertainty",
              "reason": "The post references energy markets."
            }
            """,
        ]
    )

    classification, status = classify_text(
        "Oil prices and energy policy",
        llm=lambda messages: next(responses),
    )

    assert classification.primary_topic.value == "oil_energy"
    assert status == "ok"


def test_classify_text_falls_back_after_invalid_responses() -> None:
    classification, status = classify_text(
        "China tariffs",
        llm=lambda messages: '{"primary_topic": "invalid"}',
    )

    assert classification.primary_topic.value == "other"
    assert status == "invalid_output"


def test_classify_text_falls_back_when_llm_raises() -> None:
    def broken_llm(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("transient service failure")

    classification, status = classify_text("China tariffs", llm=broken_llm)

    assert classification.primary_topic.value == "other"
    assert status == "llm_error"


def test_classify_dataframe_writes_expected_columns() -> None:
    df = pd.DataFrame({"post_id": ["1"], "cleaned_text": [""]})

    result = classify_dataframe(df, llm=None)

    assert result.loc[0, "primary_topic"] == "other"
    assert result.loc[0, "tone"] == "neutral"
    assert result.loc[0, "entities"] == ""
    assert result.loc[0, "market_relevance"] == "low"
    assert result.loc[0, "policy_direction"] == "neutral"
    assert result.loc[0, "classification_status"] == "empty_text"
    assert bool(result.loc[0, "classification_fallback"]) is True


def test_classify_dataframe_raises_if_input_already_has_classification_columns() -> None:
    df = pd.DataFrame({"post_id": ["1"], "cleaned_text": ["hello"], "primary_topic": ["other"]})

    with pytest.raises(ValueError, match="already contains classification columns"):
        classify_dataframe(df, llm=None)


def test_classify_dataframe_incremental_resumes_existing_output(tmp_path) -> None:
    output_path = tmp_path / "classified.parquet"
    df = pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "cleaned_text": ["Already paid for", "Needs classification"],
        }
    )
    existing = df.head(1).assign(
        primary_topic="crypto",
        tone="neutral",
        entities="Bitcoin",
        market_relevance="medium",
        policy_direction="neutral",
        classification_reason="Existing output.",
        classification_status="ok",
        classification_fallback=False,
        classification_text_truncated=False,
    )
    existing.to_parquet(output_path, index=False)

    result = classify_dataframe_incremental(
        df,
        llm=None,
        output_path=output_path,
        resume=True,
        show_progress=False,
    )

    assert result.loc[0, "primary_topic"] == "crypto"
    assert result.loc[1, "classification_status"] == "no_llm"
