import json

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    analyze_similar_events,
    build_summary,
    ensure_selected_ticker_columns,
    fetch_events_by_post_id,
    json_safe,
    market_reaction_summary,
    parse_ticker_list,
    selected_ticker_union,
    selected_topic_counts,
    use_retrieved_document_text,
)


def test_parse_ticker_list_handles_empty_and_comma_joined_values() -> None:
    assert parse_ticker_list("SP500, QQQ") == ["SP500", "QQQ"]
    assert parse_ticker_list("") == []
    assert parse_ticker_list(None) == []


def test_selected_ticker_union_preserves_retrieval_order() -> None:
    events = pd.DataFrame({"selected_tickers": ["SP500,QQQ", "QQQ,GLD"]})

    assert selected_ticker_union(events) == ["SP500", "QQQ", "GLD"]


def test_selected_topic_counts_sorts_by_frequency() -> None:
    events = pd.DataFrame({"primary_topic": ["other", "tariff_trade", "other"]})

    assert selected_topic_counts(events) == [
        {"primary_topic": "other", "count": 2},
        {"primary_topic": "tariff_trade", "count": 1},
    ]


def test_market_reaction_summary_uses_only_rows_where_ticker_is_selected() -> None:
    events = pd.DataFrame(
        {
            "selected_tickers": ["SP500,QQQ", "GLD"],
            "sp500_daily_return": [0.01, 0.99],
            "qqq_daily_return": [0.03, 0.99],
            "gld_daily_return": [0.99, -0.02],
        }
    )

    result = market_reaction_summary(events)

    assert result[0]["ticker"] == "SP500"
    assert result[0]["avg_daily_return"] == 0.01
    assert result[1]["ticker"] == "QQQ"
    assert result[1]["median_daily_return"] == 0.03
    assert result[2]["ticker"] == "GLD"
    assert result[2]["avg_daily_return"] == -0.02


def test_fetch_events_by_post_id_joins_retrieval_results_with_duckdb(tmp_path) -> None:
    pytest.importorskip("duckdb")
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "cleaned_text": ["China tariffs", "Oil policy"],
            "primary_topic": ["tariff_trade", "oil_energy"],
            "selected_tickers": ["SP500,QQQ", "USO"],
            "selected_return_columns": ["sp500_daily_return,qqq_daily_return", "uso_daily_return"],
            "sp500_daily_return": [0.01, 0.02],
            "qqq_daily_return": [0.03, 0.04],
            "uso_daily_return": [0.05, 0.06],
        }
    ).to_parquet(events_path, index=False)
    search_results = [
        {"post_id": "2", "score": 0.9, "distance": 0.1, "cleaned_text": "Oil policy"},
        {"post_id": "1", "score": 0.8, "distance": 0.2, "cleaned_text": "China tariffs"},
    ]

    result = fetch_events_by_post_id(search_results, events_path=events_path)

    assert result["post_id"].tolist() == ["2", "1"]
    assert result["retrieval_rank"].tolist() == [1, 2]
    assert result.loc[0, "similarity_score"] == 0.9


def test_use_retrieved_document_text_updates_display_text() -> None:
    events = pd.DataFrame(
        {
            "cleaned_text": ["RT @realDonaldTrumpOil and gas prices are moving", "China tariffs"],
            "retrieved_document_text": ["Oil and gas prices are moving", ""],
        }
    )

    result = use_retrieved_document_text(events)

    assert result["cleaned_text"].tolist() == ["Oil and gas prices are moving", "China tariffs"]


def test_ensure_selected_ticker_columns_fills_missing_mapping_columns() -> None:
    events = pd.DataFrame({"primary_topic": ["war_defense"]})

    result = ensure_selected_ticker_columns(events)

    assert result.loc[0, "selected_tickers"] == "LMT,WAR,GLD"
    assert result.loc[0, "selected_return_columns"] == (
        "lmt_daily_return,war_daily_return,gld_daily_return"
    )


def test_analyze_similar_events_returns_joined_posts_and_market_reactions(tmp_path) -> None:
    pytest.importorskip("duckdb")
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "post_id": ["1", "2"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "datetime": pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-02T12:00:00Z"]),
            "cleaned_text": ["China tariffs", "China tariff threat"],
            "primary_topic": ["tariff_trade", "tariff_trade"],
            "tone": ["threatening", "aggressive"],
            "entities": ["China,Tariffs", "China"],
            "market_relevance": ["high", "high"],
            "policy_direction": ["escalation", "escalation"],
            "selected_tickers": ["SP500,QQQ,FXI,UUP,TLT", "SP500,QQQ,FXI,UUP,TLT"],
            "selected_return_columns": [
                "sp500_daily_return,qqq_daily_return,fxi_daily_return,uup_daily_return,tlt_daily_return",
                "sp500_daily_return,qqq_daily_return,fxi_daily_return,uup_daily_return,tlt_daily_return",
            ],
            "sp500_daily_return": [0.01, 0.03],
            "qqq_daily_return": [0.02, 0.04],
            "fxi_daily_return": [-0.01, -0.03],
            "uup_daily_return": [0.001, 0.002],
            "tlt_daily_return": [-0.002, -0.004],
        }
    ).to_parquet(events_path, index=False)

    def fake_search(**kwargs):
        return [
            {"post_id": "2", "score": 0.95, "distance": 0.05},
            {"post_id": "1", "score": 0.90, "distance": 0.10},
        ]

    result = analyze_similar_events(
        "China tariff threats",
        top_k=2,
        events_path=events_path,
        search_fn=fake_search,
    )

    assert result["query_type"] == "similar_event_analysis"
    assert result["retrieved_count"] == 2
    assert result["similar_posts"][0]["post_id"] == "2"
    assert result["selected_topics"] == [{"primary_topic": "tariff_trade", "count": 2}]
    assert result["selected_tickers"] == ["SP500", "QQQ", "FXI", "UUP", "TLT"]
    assert result["market_reaction"][0]["ticker"] == "SP500"
    assert result["market_reaction"][0]["avg_daily_return"] == 0.02
    assert result["similar_posts"] is not result["retrieved_post_table"]
    assert result["similar_posts"][0]["policy_direction"] == "escalation"
    assert "sp500_daily_return" not in result["similar_posts"][0]
    assert "sp500_daily_return" in result["retrieved_post_table"][0]


def test_json_safe_serializes_timezone_aware_timestamps_as_utc() -> None:
    timestamp = pd.Timestamp("2026-01-01T04:00:00-05:00")

    assert json_safe(timestamp) == "2026-01-01T09:00:00+00:00"


def test_analyze_similar_events_result_is_json_serializable(tmp_path) -> None:
    pytest.importorskip("duckdb")
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "post_id": ["1"],
            "date": pd.to_datetime(["2026-01-01"]),
            "datetime": pd.to_datetime(["2026-01-01T12:00:00Z"]),
            "cleaned_text": ["China tariffs"],
            "primary_topic": ["tariff_trade"],
            "policy_direction": ["escalation"],
            "selected_tickers": ["SP500,QQQ"],
            "sp500_daily_return": [np.float64(0.01)],
            "qqq_daily_return": [np.float64(0.02)],
            "total_engagement": [np.int64(10)],
        }
    ).to_parquet(events_path, index=False)

    result = analyze_similar_events(
        "China tariffs",
        top_k=1,
        events_path=events_path,
        search_fn=lambda **kwargs: [
            {"post_id": "1", "score": np.float64(0.9), "distance": np.float64(0.1)}
        ],
    )

    json.dumps(result)


def test_analyze_similar_events_empty_search_shape(tmp_path) -> None:
    pytest.importorskip("duckdb")
    events_path = tmp_path / "events.parquet"
    pd.DataFrame({"post_id": ["1"], "cleaned_text": ["hello"]}).to_parquet(events_path, index=False)

    result = analyze_similar_events(
        "nothing",
        top_k=1,
        events_path=events_path,
        search_fn=lambda **kwargs: [],
    )

    assert result["retrieved_count"] == 0
    assert result["selected_topics"] == []
    assert result["selected_tickers"] == []
    assert result["similar_posts"] == []
    assert result["market_reaction"] == []
    assert result["summary"] == 'No similar posts were retrieved for "nothing".'


def test_analyze_similar_events_validates_query_and_top_k(tmp_path) -> None:
    with pytest.raises(ValueError, match="Search query"):
        analyze_similar_events("", events_path=tmp_path / "missing.parquet", search_fn=lambda **kwargs: [])
    with pytest.raises(ValueError, match="at least 1"):
        analyze_similar_events("hello", top_k=0, events_path=tmp_path / "missing.parquet", search_fn=lambda **kwargs: [])
    with pytest.raises(ValueError, match="at most"):
        analyze_similar_events("hello", top_k=201, events_path=tmp_path / "missing.parquet", search_fn=lambda **kwargs: [])


def test_build_summary_template() -> None:
    events = pd.DataFrame({"primary_topic": ["other", "other"]})
    reactions = [{"ticker": "SP500"}, {"ticker": "QQQ"}]

    assert build_summary("query", events, reactions) == (
        'Retrieved 2 similar posts for "query". Topics: other (2). '
        "Market reaction is summarized from daily open-to-close returns for: SP500, QQQ."
    )


def test_json_safe_serializes_numpy_scalars() -> None:
    assert json_safe(np.int64(3)) == 3
    assert json_safe(np.float64(1.5)) == 1.5
    assert json_safe(np.float64(np.nan)) is None
    assert json_safe(np.bool_(True)) is True
