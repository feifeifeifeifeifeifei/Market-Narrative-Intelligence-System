import warnings

import numpy as np
import pandas as pd

from src.features import add_daily_returns, add_engagement_features, add_text_features


def test_add_daily_returns_uses_open_to_close_return() -> None:
    df = pd.DataFrame({"qqq_open": [100.0], "qqq_close": [105.0]})

    result = add_daily_returns(df, ["qqq"])

    assert result.loc[0, "qqq_daily_return"] == 0.05


def test_add_daily_returns_ignores_zero_open() -> None:
    df = pd.DataFrame({"qqq_open": [0.0], "qqq_close": [105.0]})

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = add_daily_returns(df, ["qqq"])

    assert np.isnan(result.loc[0, "qqq_daily_return"])
    assert not [
        warning
        for warning in captured_warnings
        if issubclass(warning.category, RuntimeWarning)
    ]


def test_add_text_features_measures_original_text_length_before_url_stripping() -> None:
    df = pd.DataFrame(
        {
            "text": ["https://example.com"],
            "content_html": [""],
            "cleaned_text": [""],
            "has_media": [False],
        }
    )

    result = add_text_features(df)

    assert result.loc[0, "text_length"] == len("https://example.com")


def test_add_engagement_features_adds_total_and_log_engagement() -> None:
    df = pd.DataFrame(
        {
            "replies_count": [1],
            "reblogs_count": [2],
            "favourites_count": [3],
        }
    )

    result = add_engagement_features(df)

    assert result.loc[0, "total_engagement"] == 6
    assert result.loc[0, "log_engagement"] == np.log1p(6)
