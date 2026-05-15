from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

from src.clean import build_cleaned_text
from src.config import CATEGORY_COLUMNS, CLEANED_EVENTS_PATH, CORE_COLUMNS, RAW_DATA_PATH, TICKERS
from src.features import add_daily_returns, add_engagement_features, add_text_features


TIMEZONE_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)
SOURCE_TIMEZONE = "America/New_York"
OUTPUT_TIMEZONE = "UTC"


def open_close_columns(columns: pd.Index, tickers: list[str]) -> list[str]:
    selected: list[str] = []
    for ticker in tickers:
        for suffix in ("open", "close"):
            column = f"{ticker}_{suffix}"
            if column in columns:
                selected.append(column)
    return selected


def category_columns(columns: pd.Index) -> list[str]:
    return available_columns(columns, CATEGORY_COLUMNS)


def available_columns(columns: pd.Index, requested: list[str]) -> list[str]:
    return [column for column in requested if column in columns]


def enforce_unique_post_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        raise ValueError("Raw data must include a post_id column.")
    if df["post_id"].isna().any():
        raise ValueError("Raw data contains rows with missing post_id values.")
    duplicates = df["post_id"].astype(str).duplicated()
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(f"Raw data contains {duplicate_count} duplicate post_id values.")
    return df


def bool_series(index: pd.Index, value: bool) -> pd.Series:
    return pd.Series(value, index=index)


def text_empty_mask(df: pd.DataFrame) -> pd.Series:
    if "text" not in df.columns:
        return bool_series(df.index, True)
    return df["text"].fillna("").astype(str).str.strip().eq("")


def drop_empty_text_posts(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~text_empty_mask(df)].copy()


def parse_datetime_column(values: pd.Series) -> pd.Series:
    text_values = values.astype("string")
    has_explicit_timezone = text_values.str.strip().str.contains(TIMEZONE_PATTERN, na=False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")

    if has_explicit_timezone.any():
        parsed.loc[has_explicit_timezone] = pd.to_datetime(
            values.loc[has_explicit_timezone],
            errors="coerce",
            utc=True,
        )

    naive_mask = ~has_explicit_timezone
    if naive_mask.any():
        naive = pd.to_datetime(values.loc[naive_mask], errors="coerce")
        localized = naive.dt.tz_localize(
            SOURCE_TIMEZONE,
            ambiguous="NaT",
            nonexistent="NaT",
        ).dt.tz_convert(OUTPUT_TIMEZONE)
        dropped_count = int((localized.isna() & naive.notna()).sum())
        if dropped_count:
            warnings.warn(
                f"{dropped_count} naive datetime values could not be localized "
                f"to {SOURCE_TIMEZONE} and were set to NaT.",
                stacklevel=2,
            )
        parsed.loc[naive_mask] = localized

    return parsed


def parse_date_column(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def clean_events(raw_df: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or TICKERS
    enforce_unique_post_ids(raw_df)

    selected_columns = (
        available_columns(raw_df.columns, CORE_COLUMNS)
        + category_columns(raw_df.columns)
        + open_close_columns(raw_df.columns, tickers)
    )
    selected_columns = list(dict.fromkeys(selected_columns))
    df = raw_df.loc[:, selected_columns].copy()

    df = drop_empty_text_posts(df)
    df["post_id"] = df["post_id"].astype(str)
    if "datetime" in df.columns:
        df["datetime"] = parse_datetime_column(df["datetime"])
    if "date" in df.columns:
        df["date"] = parse_date_column(df["date"])

    text = df["text"] if "text" in df.columns else pd.Series([""] * len(df), index=df.index)
    content_html = (
        df["content_html"]
        if "content_html" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    df["cleaned_text"] = [
        build_cleaned_text(text_value, html_value)
        for text_value, html_value in zip(text, content_html)
    ]

    df = add_engagement_features(df)
    df = add_text_features(df)
    df = add_daily_returns(df, tickers)
    return df.reset_index(drop=True)


def run_etl(raw_path: Path = RAW_DATA_PATH, output_path: Path = CLEANED_EVENTS_PATH) -> pd.DataFrame:
    raw_df = pd.read_parquet(raw_path)
    cleaned_df = clean_events(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(output_path, index=False)
    return cleaned_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned market narrative events.")
    parser.add_argument("--raw-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=CLEANED_EVENTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_df = run_etl(args.raw_path, args.output_path)
    print(f"Wrote {len(cleaned_df):,} rows to {args.output_path}")


if __name__ == "__main__":
    main()
