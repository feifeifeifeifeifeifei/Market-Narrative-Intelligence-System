from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CLASSIFIED_EVENTS_PATH, TICKERS
from src.semantic_search import search_similar_posts
from src.ticker_mapping import return_column_for_ticker, selected_tickers_for_topic


SearchFunction = Callable[..., list[dict[str, Any]]]
MAX_TOP_K = 200

SIMILAR_POST_COLUMNS = [
    "retrieval_rank",
    "similarity_score",
    "similarity_distance",
    "post_id",
    "date",
    "datetime",
    "cleaned_text",
    "url",
    "primary_topic",
    "tone",
    "entities",
    "market_relevance",
    "policy_direction",
    "selected_tickers",
    "selected_return_columns",
    "total_engagement",
    "classification_status",
]
RETURN_COLUMNS = [return_column_for_ticker(ticker) for ticker in TICKERS]
RETRIEVED_TABLE_COLUMNS = [*SIMILAR_POST_COLUMNS, *RETURN_COLUMNS]


def require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return duckdb


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def retrieval_dataframe(search_results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "post_id": str(result["post_id"]),
                "retrieval_rank": index + 1,
                "similarity_score": result.get("score"),
                "similarity_distance": result.get("distance"),
                "retrieved_document_text": result.get("cleaned_text"),
            }
            for index, result in enumerate(search_results)
        ]
    )


def fetch_events_by_post_id(
    search_results: list[dict[str, Any]],
    events_path: Path = CLASSIFIED_EVENTS_PATH,
) -> pd.DataFrame:
    if not search_results:
        return pd.DataFrame(columns=RETRIEVED_TABLE_COLUMNS)

    duckdb = require_duckdb()
    retrieval = retrieval_dataframe(search_results)
    path_sql = quote_sql_string(str(events_path))

    with duckdb.connect(database=":memory:") as connection:
        connection.register("retrieval", retrieval)
        events = connection.execute(
            f"""
            WITH ranked AS (
                SELECT post_id, retrieval_rank, similarity_score, similarity_distance, retrieved_document_text
                FROM retrieval
            )
            SELECT
                ranked.retrieval_rank,
                ranked.similarity_score,
                ranked.similarity_distance,
                ranked.retrieved_document_text,
                events.*
            FROM ranked
            LEFT JOIN read_parquet({path_sql}) AS events
              ON CAST(events.post_id AS VARCHAR) = ranked.post_id
            WHERE CAST(events.post_id AS VARCHAR) IN (SELECT post_id FROM ranked)
            ORDER BY ranked.retrieval_rank
            """
        ).fetchdf()
    return ensure_selected_ticker_columns(use_retrieved_document_text(events))


def use_retrieved_document_text(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    if "retrieved_document_text" not in result.columns or "cleaned_text" not in result.columns:
        return result

    retrieved_text = result["retrieved_document_text"].map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    result.loc[retrieved_text != "", "cleaned_text"] = retrieved_text.loc[retrieved_text != ""]
    return result


def ensure_selected_ticker_columns(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    if "selected_tickers" not in result.columns:
        result["selected_tickers"] = result["primary_topic"].map(
            lambda topic: ",".join(selected_tickers_for_topic(topic))
        )
    if "selected_return_columns" not in result.columns:
        result["selected_return_columns"] = result["selected_tickers"].map(
            lambda value: ",".join(return_column_for_ticker(ticker) for ticker in parse_ticker_list(value))
        )
    return result


def parse_ticker_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [ticker.strip() for ticker in str(value).split(",") if ticker.strip()]


def selected_ticker_union(events: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for value in events.get("selected_tickers", pd.Series(dtype=str)).fillna(""):
        for ticker in parse_ticker_list(value):
            if ticker not in selected:
                selected.append(ticker)
    return selected


def selected_topic_counts(events: pd.DataFrame) -> list[dict[str, Any]]:
    if "primary_topic" not in events.columns or events.empty:
        return []
    counts = Counter(
        topic
        for topic in events["primary_topic"].fillna("").astype(str)
        if topic
    )
    return [
        {"primary_topic": topic, "count": count}
        for topic, count in counts.most_common()
    ]


def market_reaction_summary(events: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for ticker in selected_ticker_union(events):
        return_column = return_column_for_ticker(ticker)
        if return_column not in events.columns:
            continue

        selected_mask = events["selected_tickers"].fillna("").map(
            lambda value, selected_ticker=ticker: selected_ticker in parse_ticker_list(value)
        )
        values = pd.to_numeric(events.loc[selected_mask, return_column], errors="coerce").dropna()
        summaries.append(
            {
                "ticker": ticker,
                "return_column": return_column,
                "avg_daily_return": safe_float(values.mean()),
                "median_daily_return": safe_float(values.median()),
                "min_daily_return": safe_float(values.min()),
                "max_daily_return": safe_float(values.max()),
                "sample_size": int(values.count()),
            }
        )
    return summaries


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def similar_posts_table(events: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [column for column in SIMILAR_POST_COLUMNS if column in events.columns]
    return dataframe_records(events.loc[:, columns].copy())


def retrieved_post_table(events: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [column for column in RETRIEVED_TABLE_COLUMNS if column in events.columns]
    return dataframe_records(events.loc[:, columns].copy())


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return [json_safe_record(record) for record in records]


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: json_safe(value) for key, value in record.items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        return value.isoformat()
    if is_numpy_bool(value):
        return bool(value)
    if is_numpy_integer(value):
        return int(value)
    if is_numpy_floating(value):
        float_value = float(value)
        return None if math.isnan(float_value) else float_value
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def is_numpy_bool(value: Any) -> bool:
    return value.__class__.__module__ == "numpy" and value.__class__.__name__ == "bool_"


def is_numpy_integer(value: Any) -> bool:
    return value.__class__.__module__ == "numpy" and "int" in value.__class__.__name__


def is_numpy_floating(value: Any) -> bool:
    return value.__class__.__module__ == "numpy" and "float" in value.__class__.__name__


def build_summary(query: str, events: pd.DataFrame, market_reaction: list[dict[str, Any]]) -> str:
    """M5 deterministic summary; M11 can swap in an LLM grounded on this result dict."""
    if events.empty:
        return f'No similar posts were retrieved for "{query}".'

    topic_text = ", ".join(
        f"{item['primary_topic']} ({item['count']})" for item in selected_topic_counts(events)
    )
    tickers = ", ".join(item["ticker"] for item in market_reaction) or "no selected tickers"
    return (
        f'Retrieved {len(events)} similar posts for "{query}". '
        f"Topics: {topic_text or 'none'}. "
        f"Market reaction is summarized from daily open-to-close returns for: {tickers}."
    )


def normalize_analysis_filters(filters: dict[str, str] | None) -> dict[str, str]:
    return {
        field: str(value).strip()
        for field, value in (filters or {}).items()
        if str(value).strip() and str(value).strip().lower() != "all"
    }


def analyze_similar_events(
    query: str,
    top_k: int | None = None,
    events_path: Path = CLASSIFIED_EVENTS_PATH,
    search_fn: SearchFunction = search_similar_posts,
    filters: dict[str, str] | None = None,
    **search_kwargs: Any,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if top_k is not None and top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be at most {MAX_TOP_K}.")

    normalized_filters = normalize_analysis_filters(filters)
    search_results = search_fn(query=query, top_k=top_k, filters=normalized_filters, **search_kwargs)
    events = fetch_events_by_post_id(search_results, events_path=events_path)
    reactions = market_reaction_summary(events)
    similar_posts = similar_posts_table(events)
    retrieved_table = retrieved_post_table(events)

    return {
        "query": query,
        "query_type": "similar_event_analysis",
        "top_k": top_k,
        "filters": normalized_filters,
        "retrieved_count": len(similar_posts),
        "selected_topics": selected_topic_counts(events),
        "selected_tickers": selected_ticker_union(events),
        "similar_posts": similar_posts,
        "market_reaction": reactions,
        "retrieved_post_table": retrieved_table,
        "summary": build_summary(query, events, reactions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze market reactions for similar posts.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--events-path", type=Path, default=CLASSIFIED_EVENTS_PATH)
    parser.add_argument("--embedding-provider", choices=["auto", "openai", "hashing", "local_hashing"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_similar_events(
        query=args.query,
        top_k=args.top_k,
        events_path=args.events_path,
        embedding_provider_kind=args.embedding_provider,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
