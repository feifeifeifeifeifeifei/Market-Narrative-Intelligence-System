import pandas as pd
import pytest
import src.classify as classify_module

from src.classify import (
    FatalClassificationError,
    build_batch_classification_messages,
    build_classification_messages,
    classify_dataframe,
    classify_dataframe_incremental,
    run_classification,
    classify_text_batch,
    classify_text,
    supports_zero_temperature,
    validate_batch_classification,
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


def test_classification_prompt_balances_specific_and_conservative_labels() -> None:
    messages = build_classification_messages("China tariffs and Fed policy")
    prompt = "\n".join(message["content"] for message in messages)

    assert "Prefer specific labels when supported" in prompt
    assert "use other for vague, ceremonial, or personal posts" in prompt
    assert "Use low market_relevance for" in prompt
    assert "fitness/awareness programs" in prompt
    assert "tariff_trade=tariffs" in prompt


def test_batch_classification_prompt_uses_item_ids() -> None:
    messages = build_batch_classification_messages(
        [("0", "China tariffs"), ("1", "Stock market record high")]
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert '"classifications"' in prompt
    assert '"item_id": "same item_id from input"' in prompt
    assert "item_id: 0" in prompt
    assert "item_id: 1" in prompt


def test_gpt5_models_use_default_temperature() -> None:
    assert supports_zero_temperature("gpt-4o-mini") is True
    assert supports_zero_temperature("gpt-5-mini") is False


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


def test_validate_batch_classification_accepts_valid_json() -> None:
    result = validate_batch_classification(
        """
        {
          "classifications": [
            {
              "item_id": "0",
              "primary_topic": "tariff_trade",
              "tone": "threatening",
              "entities": ["China", "Tariffs"],
              "market_relevance": "high",
              "policy_direction": "escalation",
              "reason": "The post discusses tariff threats involving China."
            },
            {
              "item_id": "1",
              "primary_topic": "broad_market",
              "tone": "praising",
              "entities": ["Stock Market"],
              "market_relevance": "high",
              "policy_direction": "neutral",
              "reason": "The post praises stock market performance."
            }
          ]
        }
        """,
        expected_item_ids={"0", "1"},
    )

    assert result["0"].primary_topic.value == "tariff_trade"
    assert result["1"].primary_topic.value == "broad_market"


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


def test_classify_text_batch_classifies_multiple_posts_in_one_call() -> None:
    calls = []

    def llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return """
        {
          "classifications": [
            {
              "item_id": "0",
              "primary_topic": "tariff_trade",
              "tone": "threatening",
              "entities": ["China"],
              "market_relevance": "high",
              "policy_direction": "escalation",
              "reason": "The post references China tariffs."
            },
            {
              "item_id": "1",
              "primary_topic": "broad_market",
              "tone": "praising",
              "entities": ["Stock Market"],
              "market_relevance": "high",
              "policy_direction": "neutral",
              "reason": "The post references market performance."
            }
          ]
        }
        """

    results = classify_text_batch(["China tariffs", "Stock market high"], llm=llm)

    assert len(calls) == 1
    assert [status for _, status in results] == ["ok", "ok"]
    assert results[0][0].primary_topic.value == "tariff_trade"
    assert results[1][0].primary_topic.value == "broad_market"


def test_classify_text_batch_handles_empty_text_without_llm() -> None:
    results = classify_text_batch([""], llm=lambda messages: pytest.fail("llm should not be called"))

    assert results[0][0].primary_topic.value == "other"
    assert results[0][1] == "empty_text"


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


def test_classify_text_propagates_fatal_llm_errors() -> None:
    def broken_llm(messages: list[dict[str, str]]) -> str:
        raise FatalClassificationError("quota unavailable")

    with pytest.raises(FatalClassificationError, match="quota unavailable"):
        classify_text("China tariffs", llm=broken_llm)


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


def test_run_classification_requires_live_llm_unless_fallback_only(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "cleaned.parquet"
    output_path = tmp_path / "classified.parquet"
    pd.DataFrame({"post_id": ["1"], "cleaned_text": ["China tariffs"]}).to_parquet(
        input_path,
        index=False,
    )
    monkeypatch.setattr(classify_module, "create_openai_llm", lambda model: None)
    monkeypatch.setattr(
        classify_module,
        "openai_llm_unavailable_reason",
        lambda: "test unavailable.",
    )

    with pytest.raises(RuntimeError, match="OpenAI live classification is unavailable"):
        run_classification(input_path=input_path, output_path=output_path)

    result = run_classification(
        input_path=input_path,
        output_path=output_path,
        fallback_only=True,
        progress=False,
    )

    assert result.loc[0, "classification_status"] == "no_llm"
