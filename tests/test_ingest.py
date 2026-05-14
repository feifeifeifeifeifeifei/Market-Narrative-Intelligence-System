import pandas as pd
import pytest

from src.ingest import (
    category_columns,
    clean_events,
    drop_empty_non_media_posts,
    open_close_columns,
    parse_datetime_column,
)


def test_open_close_columns_excludes_intraday_market_columns() -> None:
    columns = pd.Index(["qqq_open", "qqq_close", "qqq_5min_after", "qqq_1hr_before"])

    assert open_close_columns(columns, ["qqq"]) == ["qqq_open", "qqq_close"]


def test_category_columns_only_keeps_known_classification_columns() -> None:
    columns = pd.Index(["cat_attacking_individual", "cat_egory_future", "gdelt_military"])

    assert category_columns(columns) == ["cat_attacking_individual", "gdelt_military"]


def test_parse_datetime_column_localizes_naive_values_as_new_york() -> None:
    parsed = parse_datetime_column(
        pd.Series(["2026-01-01 00:30:00", "2026-01-01T00:30:00Z"])
    )

    assert parsed.iloc[0] == pd.Timestamp("2026-01-01T05:30:00Z")
    assert parsed.iloc[1] == pd.Timestamp("2026-01-01T00:30:00Z")


def test_parse_datetime_column_warns_when_dst_ambiguity_becomes_nat() -> None:
    with pytest.warns(UserWarning, match="could not be localized"):
        parsed = parse_datetime_column(pd.Series(["2026-11-01 01:30:00"]))

    assert pd.isna(parsed.iloc[0])


def test_drop_empty_non_media_posts_handles_missing_optional_columns() -> None:
    raw = pd.DataFrame({"post_id": ["1", "2"], "text": ["", "hello"]})

    result = drop_empty_non_media_posts(raw)

    assert result["post_id"].tolist() == ["2"]


def test_clean_events_rejects_duplicate_post_ids() -> None:
    raw = pd.DataFrame(
        {
            "post_id": ["1", "1"],
            "datetime": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
            "date": ["2026-01-01", "2026-01-01"],
            "text": ["one", "two"],
            "content_html": ["", ""],
            "has_media": [False, False],
            "replies_count": [0, 0],
            "reblogs_count": [0, 0],
            "favourites_count": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="duplicate post_id"):
        clean_events(raw, tickers=["qqq"])


def test_clean_events_rejects_duplicate_post_ids_after_string_cast() -> None:
    raw = pd.DataFrame(
        {
            "post_id": [1, "1"],
            "datetime": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
            "date": ["2026-01-01", "2026-01-01"],
            "text": ["one", "two"],
            "content_html": ["", ""],
            "has_media": [False, False],
            "replies_count": [0, 0],
            "reblogs_count": [0, 0],
            "favourites_count": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="duplicate post_id"):
        clean_events(raw, tickers=["qqq"])


def test_clean_events_keeps_date_as_datetime_column() -> None:
    raw = pd.DataFrame(
        {
            "post_id": ["1"],
            "datetime": ["2026-01-01T00:00:00Z"],
            "date": ["2026-01-01"],
            "text": ["hello"],
            "content_html": [""],
            "has_media": [False],
            "replies_count": [0],
            "reblogs_count": [0],
            "favourites_count": [0],
        }
    )

    result = clean_events(raw, tickers=["qqq"])

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
