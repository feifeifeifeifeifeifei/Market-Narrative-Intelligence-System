from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics import RETRIEVED_TABLE_COLUMNS, analyze_similar_events
from src.build_chroma import get_chroma_client, get_collection
from src.config import CLASSIFIED_EVENTS_PATH, PROJECT_ROOT, TICKERS
from src.export_powerbi import write_csv_atomic
from src.ticker_mapping import return_column_for_ticker


logger = logging.getLogger(__name__)
REPORTS_DIR = PROJECT_ROOT / "reports"
POWERBI_REPORT_DIR = REPORTS_DIR / "powerbi"
POWERBI_TABLES_DIR = POWERBI_REPORT_DIR / "tables"
POWERBI_SCREENSHOTS_DIR = POWERBI_REPORT_DIR / "screenshots"
RESUME_BULLETS_PATH = REPORTS_DIR / "resume_bullets.md"
KNOWN_CLASSIFICATION_STATUSES = ("ok", "empty_text", "no_llm", "invalid_output", "llm_error", "not_processed")


def ensure_report_dirs() -> None:
    POWERBI_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_events(path: Path = CLASSIFIED_EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def narrative_topic_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("primary_topic", dropna=False)
        .size()
        .reset_index(name="post_count")
        .sort_values("post_count", ascending=False)
    )


def posts_over_time(df: pd.DataFrame) -> pd.DataFrame:
    by_week = df.dropna(subset=["date"]).copy()
    by_week["week_start"] = by_week["date"].dt.to_period("W").dt.start_time
    return (
        by_week.groupby("week_start")
        .size()
        .reset_index(name="post_count")
        .sort_values("week_start")
    )


def tone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["primary_topic", "tone"], dropna=False)
        .size()
        .reset_index(name="post_count")
        .sort_values(["primary_topic", "post_count"], ascending=[True, False])
    )


def policy_direction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "policy_direction" not in df.columns:
        return pd.DataFrame(columns=["primary_topic", "policy_direction", "post_count"])
    return (
        df.groupby(["primary_topic", "policy_direction"], dropna=False)
        .size()
        .reset_index(name="post_count")
        .sort_values(["primary_topic", "post_count"], ascending=[True, False])
    )


def market_reaction_by_topic_ticker(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic, topic_df in df.groupby("primary_topic", dropna=False):
        for ticker in TICKERS:
            return_col = return_column_for_ticker(ticker)
            if return_col not in topic_df.columns:
                continue
            returns = pd.to_numeric(topic_df[return_col], errors="coerce").dropna()
            rows.append(
                {
                    "primary_topic": topic,
                    "ticker": ticker.upper(),
                    "avg_daily_return": safe_float(returns.mean()),
                    "median_daily_return": safe_float(returns.median()),
                    "sample_size": int(returns.count()),
                    "missing_return_count": int(topic_df[return_col].isna().sum()),
                    "missing_return_rate": safe_float(topic_df[return_col].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def selected_ticker_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "selected_tickers" not in df.columns or "primary_topic" not in df.columns:
        return pd.DataFrame(columns=["primary_topic", "ticker", "post_count"])

    exploded = (
        df[["primary_topic", "selected_tickers"]]
        .fillna("")
        .assign(ticker=lambda frame: frame["selected_tickers"].str.split(","))
        .explode("ticker")
    )
    exploded["ticker"] = exploded["ticker"].fillna("").astype(str).str.strip()
    exploded = exploded.loc[exploded["ticker"] != ""]
    if exploded.empty:
        return pd.DataFrame(columns=["primary_topic", "ticker", "post_count"])
    return (
        exploded
        .groupby(["primary_topic", "ticker"])
        .size()
        .reset_index(name="post_count")
        .sort_values(["primary_topic", "post_count"], ascending=[True, False])
    )


def high_engagement_posts(df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    columns = [
        "post_id",
        "date",
        "cleaned_text",
        "url",
        "primary_topic",
        "tone",
        "policy_direction",
        "market_relevance",
        "selected_tickers",
        "total_engagement",
        "sp500_daily_return",
        "qqq_daily_return",
        "gld_daily_return",
        "tlt_daily_return",
        "uso_daily_return",
    ]
    available = [column for column in columns if column in df.columns]
    result = df.loc[:, available].copy()
    if "total_engagement" in result.columns:
        result = result.sort_values("total_engagement", ascending=False)
    return result.head(limit).reset_index(drop=True)


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    fallback_count = int(df.get("classification_fallback", pd.Series(dtype=bool)).fillna(False).sum())
    status_counts = (
        df.get("classification_status", pd.Series(dtype=str))
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict()
    )
    rows = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "classification_fallback_count", "value": fallback_count},
        {"metric": "classification_fallback_rate", "value": safe_ratio(fallback_count, total_rows)},
        {
            "metric": "empty_cleaned_text_count",
            "value": int(df["cleaned_text"].fillna("").astype(str).str.strip().eq("").sum()),
        },
    ]
    for status in KNOWN_CLASSIFICATION_STATUSES:
        rows.append({"metric": f"classification_{status}_count", "value": int(status_counts.get(status, 0))})
    for ticker in TICKERS:
        return_col = return_column_for_ticker(ticker)
        if return_col in df.columns:
            missing_count = int(df[return_col].isna().sum())
            rows.append({"metric": f"{ticker}_missing_return_count", "value": missing_count})
            rows.append({"metric": f"{ticker}_missing_return_rate", "value": safe_ratio(missing_count, total_rows)})
    return pd.DataFrame(rows)


def narrative_analysis_events(df: pd.DataFrame) -> pd.DataFrame:
    if "classification_status" not in df.columns:
        return df
    return df.loc[df["classification_status"] == "ok"].copy()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def similar_event_sample(query: str, top_k: int, embedding_provider: str) -> pd.DataFrame:
    try:
        result = analyze_similar_events(
            query=query,
            top_k=top_k,
            embedding_provider_kind=embedding_provider,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.warning("similar_event_sample skipped: %s", exc)
        return empty_similar_event_sample()
    rows = result["retrieved_post_table"]
    if not rows:
        return empty_similar_event_sample()
    table = pd.DataFrame(rows)
    table.insert(0, "query", query)
    return table


def empty_similar_event_sample() -> pd.DataFrame:
    return pd.DataFrame(columns=["query", *RETRIEVED_TABLE_COLUMNS])


def build_report_tables(
    df: pd.DataFrame,
    query: str = "China tariff threats",
    embedding_provider: str = "hashing",
) -> dict[str, pd.DataFrame]:
    analysis_df = narrative_analysis_events(df)
    return {
        "narrative_topic_counts": narrative_topic_counts(analysis_df),
        "posts_over_time_weekly": posts_over_time(df),
        "tone_distribution": tone_distribution(analysis_df),
        "policy_direction_distribution": policy_direction_distribution(analysis_df),
        "market_reaction_by_topic_ticker": market_reaction_by_topic_ticker(analysis_df),
        "selected_ticker_distribution": selected_ticker_distribution(analysis_df),
        "high_engagement_posts": high_engagement_posts(analysis_df),
        "data_quality_summary": data_quality_summary(df),
        "similar_event_search_output": similar_event_sample(query, top_k=20, embedding_provider=embedding_provider),
    }


def write_report_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    ensure_report_dirs()
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = POWERBI_TABLES_DIR / f"{name}.csv"
        write_csv_atomic(table, path)
        paths[name] = path
    return paths


def render_dashboard_previews(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [
        render_narrative_overview(df, tables, plt),
        render_market_reaction(df, tables, plt),
        render_similar_event_output(tables, plt),
        render_data_quality(tables, plt),
    ]
    return paths


def render_narrative_overview(df: pd.DataFrame, tables: dict[str, pd.DataFrame], plt: Any) -> Path:
    topic_counts = tables["narrative_topic_counts"]
    weekly = tables["posts_over_time_weekly"]
    tone = tables["tone_distribution"]
    path = POWERBI_SCREENSHOTS_DIR / "01_narrative_overview.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Narrative Overview", fontsize=18, fontweight="bold")
    axes[0, 0].bar(topic_counts["primary_topic"].head(8), topic_counts["post_count"].head(8), color="#2b6cb0")
    axes[0, 0].set_title("Post Count By Topic (OK Classifications)")
    axes[0, 0].tick_params(axis="x", rotation=35)
    if weekly.empty:
        axes[0, 1].text(0.05, 0.8, "No dated posts available.", fontsize=12)
    else:
        weekly.tail(80).plot(x="week_start", y="post_count", ax=axes[0, 1], color="#2f855a", legend=False)
    axes[0, 1].set_title("Posts Over Time")
    tone_totals = tone.groupby("tone")["post_count"].sum().sort_values(ascending=False).head(8) if not tone.empty else pd.Series(dtype=float)
    if tone_totals.empty:
        axes[1, 0].text(0.05, 0.8, "No tone rows available.", fontsize=12)
    else:
        tone_totals.plot(kind="bar", ax=axes[1, 0], color="#805ad5", legend=False)
    axes[1, 0].set_title("Tone Distribution (OK Classifications)")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 1].axis("off")
    date_range = date_range_text(df)
    fallback_rate = fallback_rate_text(tables["data_quality_summary"])
    ok_count = quality_metric_value(tables["data_quality_summary"], "classification_ok_count")
    axes[1, 1].text(
        0,
        0.8,
        f"Posts: {len(df):,}\nValidated LLM rows: {ok_count:,.0f}\n"
        f"Date range: {date_range}\nClassification fallback rate: {fallback_rate}",
        fontsize=13,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_market_reaction(df: pd.DataFrame, tables: dict[str, pd.DataFrame], plt: Any) -> Path:
    reactions = tables["market_reaction_by_topic_ticker"]
    selected = tables["selected_ticker_distribution"]
    path = POWERBI_SCREENSHOTS_DIR / "02_market_reaction.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Market Reaction", fontsize=18, fontweight="bold")
    overall_reactions = (
        reactions.groupby("ticker")["avg_daily_return"].mean().sort_values(ascending=False).head(12)
        if not reactions.empty
        else pd.Series(dtype=float)
    )
    axes[0].bar(overall_reactions.index, overall_reactions.values, color="#2b6cb0")
    axes[0].axhline(0, color="#2d3748", linewidth=0.8)
    axes[0].set_title("Average Daily Return By Ticker")
    axes[0].tick_params(axis="x", rotation=35)
    if selected.empty:
        axes[1].text(0.05, 0.8, "No selected ticker rows available.", fontsize=12)
        axes[1].axis("off")
    else:
        selected.head(12).plot(kind="bar", x="ticker", y="post_count", ax=axes[1], color="#dd6b20", legend=False)
        axes[1].set_title("Selected Tickers By Topic")
        axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_similar_event_output(tables: dict[str, pd.DataFrame], plt: Any) -> Path:
    sample = tables["similar_event_search_output"]
    path = POWERBI_SCREENSHOTS_DIR / "03_similar_event_search.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title("Similar Event Search Output", fontsize=18, fontweight="bold", pad=16)
    columns = [column for column in ["post_id", "similarity_score", "primary_topic", "selected_tickers", "cleaned_text"] if column in sample.columns]
    preview = sample.loc[:, columns].head(8).copy()
    if preview.empty:
        ax.text(0.05, 0.8, "No similar-event sample rows available.", fontsize=12, va="top")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path
    if "cleaned_text" in preview.columns:
        preview["cleaned_text"] = preview["cleaned_text"].fillna("").astype(str).str.slice(0, 80)
    table = ax.table(cellText=preview.values, colLabels=preview.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_data_quality(tables: dict[str, pd.DataFrame], plt: Any) -> Path:
    quality = tables["data_quality_summary"]
    path = POWERBI_SCREENSHOTS_DIR / "04_data_quality.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Data Quality", fontsize=18, fontweight="bold")
    key_metrics = quality[quality["metric"].isin(
        [
            "total_rows",
            "classification_ok_count",
            "classification_fallback_count",
            "classification_fallback_rate",
            "classification_empty_text_count",
            "classification_no_llm_count",
            "classification_invalid_output_count",
            "classification_llm_error_count",
            "empty_cleaned_text_count",
        ]
    )]
    axes[0].axis("off")
    table = axes[0].table(cellText=key_metrics.values, colLabels=key_metrics.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    axes[1].axis("off")
    missing_rates = quality[
        quality["metric"].str.endswith("_missing_return_rate")
    ].sort_values("value", ascending=False).head(6)
    missing_table = axes[1].table(
        cellText=missing_rates.values,
        colLabels=missing_rates.columns,
        loc="center",
        cellLoc="left",
    )
    missing_table.auto_set_font_size(False)
    missing_table.set_fontsize(9)
    missing_table.scale(1, 1.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def date_range_text(df: pd.DataFrame) -> str:
    if "date" not in df.columns:
        return "no dated posts"
    date_min = df["date"].min()
    date_max = df["date"].max()
    if pd.isna(date_min) or pd.isna(date_max):
        return "no dated posts"
    return f"{date_min.date()} to {date_max.date()}"


def fallback_rate_text(quality: pd.DataFrame) -> str:
    row = quality.loc[quality["metric"] == "classification_fallback_rate", "value"]
    if row.empty or pd.isna(row.iloc[0]):
        return "unknown"
    return f"{float(row.iloc[0]):.1%}"


def quality_metric_value(quality: pd.DataFrame, metric: str) -> float:
    row = quality.loc[quality["metric"] == metric, "value"]
    if row.empty or pd.isna(row.iloc[0]):
        return 0.0
    return float(row.iloc[0])


def write_powerbi_spec() -> Path:
    path = POWERBI_REPORT_DIR / "dashboard_spec.md"
    path.write_text(
        """# Power BI Dashboard Specification

## Data Sources

- `reports/powerbi/tables/narrative_topic_counts.csv`
- `reports/powerbi/tables/posts_over_time_weekly.csv`
- `reports/powerbi/tables/tone_distribution.csv`
- `reports/powerbi/tables/policy_direction_distribution.csv`
- `reports/powerbi/tables/market_reaction_by_topic_ticker.csv`
- `reports/powerbi/tables/selected_ticker_distribution.csv`
- `reports/powerbi/tables/high_engagement_posts.csv`
- `reports/powerbi/tables/similar_event_search_output.csv`
- `reports/powerbi/tables/data_quality_summary.csv`

## Pages

1. Narrative Overview: topic counts, posts over time, tone distribution, policy direction distribution.
2. Market Reaction: average/median daily returns by topic and ticker, selected ticker counts, high-engagement posts.
3. Similar Event Search Output: retrieved posts, selected tickers, similarity scores, corresponding daily returns.
4. Data Quality: classification status, fallback rate, missing return rates, empty text count.

## Current Data Note

The dashboard preview tables use rows with `classification_status = ok` for narrative,
ticker, and market-reaction views, while the Data Quality page reports the full classified
artifact including empty-text and failed LLM rows.
""",
        encoding="utf-8",
    )
    return path


def write_resume_bullets() -> Path:
    RESUME_BULLETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESUME_BULLETS_PATH.write_text(
        """# Resume Bullets

- Built an end-to-end market narrative intelligence pipeline that transforms Truth Social posts into cleaned analytical events, validates structured LLM classifications, and computes daily open-to-close asset returns across equity, rates, commodity, and crypto ETFs.
- Implemented deterministic topic-to-ticker mapping and ChromaDB semantic retrieval, enabling similar-event search joined back to DuckDB market reaction analysis by stable `post_id`.
- Produced Power BI-ready dashboard tables and preview screenshots covering narrative trends, selected asset reactions, similar-event outputs, and data quality diagnostics.
""",
        encoding="utf-8",
    )
    return RESUME_BULLETS_PATH


def build_powerbi_assets(
    events_path: Path = CLASSIFIED_EVENTS_PATH,
    query: str = "China tariff threats",
    embedding_provider: str = "hashing",
    validate_chroma: bool = True,
) -> dict[str, Any]:
    if validate_chroma:
        validate_chroma_collection_available()
    df = load_events(events_path)
    tables = build_report_tables(df, query=query, embedding_provider=embedding_provider)
    table_paths = write_report_tables(tables)
    screenshot_paths = render_dashboard_previews(df, tables)
    spec_path = write_powerbi_spec()
    resume_path = write_resume_bullets()
    return {
        "table_paths": table_paths,
        "screenshot_paths": screenshot_paths,
        "spec_path": spec_path,
        "resume_path": resume_path,
    }


def validate_chroma_collection_available() -> None:
    try:
        collection = get_collection(get_chroma_client())
        collection.count()
    except Exception as exc:
        raise RuntimeError("ChromaDB collection not found; run `python -m src.build_chroma` first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Power BI dashboard source assets.")
    parser.add_argument("--events-path", type=Path, default=CLASSIFIED_EVENTS_PATH)
    parser.add_argument("--query", default="China tariff threats")
    parser.add_argument("--embedding-provider", choices=["auto", "openai", "hashing", "local_hashing"], default="hashing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = build_powerbi_assets(
        events_path=args.events_path,
        query=args.query,
        embedding_provider=args.embedding_provider,
    )
    print(f"Wrote {len(assets['table_paths'])} dashboard tables to {POWERBI_TABLES_DIR}")
    print(f"Wrote {len(assets['screenshot_paths'])} preview screenshots to {POWERBI_SCREENSHOTS_DIR}")
    print(f"Wrote dashboard spec to {assets['spec_path']}")
    print(f"Wrote resume bullets to {assets['resume_path']}")


if __name__ == "__main__":
    main()
