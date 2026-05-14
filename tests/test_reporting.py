import pandas as pd

import src.reporting as reporting
from src.reporting import (
    build_powerbi_assets,
    build_report_tables,
    data_quality_summary,
    high_engagement_posts,
    market_reaction_by_topic_ticker,
    narrative_topic_counts,
    policy_direction_distribution,
    posts_over_time,
    selected_ticker_distribution,
    similar_event_sample,
    tone_distribution,
)


def sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-08"]),
            "cleaned_text": ["China tariffs", ""],
            "primary_topic": ["tariff_trade", "other"],
            "tone": ["threatening", "neutral"],
            "policy_direction": ["escalation", "neutral"],
            "market_relevance": ["high", "low"],
            "selected_tickers": ["SP500,QQQ", "SP500,QQQ"],
            "total_engagement": [10, 30],
            "classification_status": ["ok", "empty_text"],
            "classification_fallback": [False, True],
            "sp500_daily_return": [0.01, 0.02],
            "qqq_daily_return": [0.03, None],
        }
    )


def test_narrative_topic_counts() -> None:
    result = narrative_topic_counts(sample_events())

    assert set(result["primary_topic"]) == {"tariff_trade", "other"}
    assert result["post_count"].sum() == 2


def test_posts_over_time_groups_by_week() -> None:
    result = posts_over_time(sample_events())

    assert list(result.columns) == ["week_start", "post_count"]
    assert result["post_count"].sum() == 2


def test_tone_and_policy_direction_distributions() -> None:
    tone = tone_distribution(sample_events())
    direction = policy_direction_distribution(sample_events())

    assert {"primary_topic", "tone", "post_count"}.issubset(tone.columns)
    assert {"primary_topic", "policy_direction", "post_count"}.issubset(direction.columns)


def test_policy_direction_distribution_handles_older_parquet_shape() -> None:
    df = sample_events().drop(columns=["policy_direction"])

    result = policy_direction_distribution(df)

    assert list(result.columns) == ["primary_topic", "policy_direction", "post_count"]
    assert result.empty


def test_market_reaction_by_topic_ticker_includes_return_stats() -> None:
    result = market_reaction_by_topic_ticker(sample_events())
    row = result[(result["primary_topic"] == "tariff_trade") & (result["ticker"] == "SP500")].iloc[0]

    assert row["avg_daily_return"] == 0.01
    assert row["sample_size"] == 1


def test_selected_ticker_distribution_explodes_tickers() -> None:
    result = selected_ticker_distribution(sample_events())

    assert set(result["ticker"]) == {"SP500", "QQQ"}


def test_high_engagement_posts_sorts_descending() -> None:
    result = high_engagement_posts(sample_events(), limit=1)

    assert result.loc[0, "post_id"] == "2"


def test_high_engagement_posts_handles_missing_total_engagement() -> None:
    result = high_engagement_posts(sample_events().drop(columns=["total_engagement"]), limit=1)

    assert len(result) == 1


def test_data_quality_summary_contains_fallback_rate() -> None:
    result = data_quality_summary(sample_events())
    metrics = dict(zip(result["metric"], result["value"]))

    assert metrics["total_rows"] == 2
    assert metrics["classification_fallback_count"] == 1
    assert metrics["classification_fallback_rate"] == 0.5
    assert metrics["classification_invalid_output_count"] == 0
    assert metrics["classification_llm_error_count"] == 0


def test_similar_event_sample_keeps_schema_when_known_failure_occurs(monkeypatch) -> None:
    def broken_analyze(**kwargs):
        raise ValueError("provider mismatch with local path details")

    monkeypatch.setattr(reporting, "analyze_similar_events", broken_analyze)

    result = similar_event_sample("query", top_k=20, embedding_provider="hashing")

    assert "error" not in result.columns
    assert list(result.columns)[0] == "query"
    assert "post_id" in result.columns
    assert result.empty


def test_build_report_tables_returns_stable_table_shapes(monkeypatch) -> None:
    def fake_analyze(**kwargs):
        return {
            "retrieved_post_table": [
                {
                    "retrieval_rank": 1,
                    "similarity_score": 0.9,
                    "post_id": "1",
                    "primary_topic": "tariff_trade",
                    "selected_tickers": "SP500,QQQ",
                    "cleaned_text": "China tariffs",
                }
            ]
        }

    monkeypatch.setattr(reporting, "analyze_similar_events", fake_analyze)

    tables = build_report_tables(sample_events(), query="China tariffs", embedding_provider="hashing")

    assert set(tables) == {
        "narrative_topic_counts",
        "posts_over_time_weekly",
        "tone_distribution",
        "policy_direction_distribution",
        "market_reaction_by_topic_ticker",
        "selected_ticker_distribution",
        "high_engagement_posts",
        "data_quality_summary",
        "similar_event_search_output",
    }
    assert "query" in tables["similar_event_search_output"].columns


def test_build_powerbi_assets_writes_tables_screenshots_and_docs(tmp_path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporting, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(reporting, "POWERBI_REPORT_DIR", reports_dir / "powerbi")
    monkeypatch.setattr(reporting, "POWERBI_TABLES_DIR", reports_dir / "powerbi" / "tables")
    monkeypatch.setattr(reporting, "POWERBI_SCREENSHOTS_DIR", reports_dir / "powerbi" / "screenshots")
    monkeypatch.setattr(reporting, "RESUME_BULLETS_PATH", reports_dir / "resume_bullets.md")

    def fake_analyze(**kwargs):
        return {"retrieved_post_table": []}

    monkeypatch.setattr(reporting, "analyze_similar_events", fake_analyze)
    events_path = tmp_path / "events.parquet"
    sample_events().to_parquet(events_path, index=False)

    assets = build_powerbi_assets(events_path=events_path, validate_chroma=False)

    assert len(assets["table_paths"]) == 9
    assert len(assets["screenshot_paths"]) == 4
    assert all(path.exists() for path in assets["table_paths"].values())
    assert all(path.exists() and path.stat().st_size > 0 for path in assets["screenshot_paths"])
    assert assets["spec_path"].exists()
    assert assets["resume_path"].exists()
