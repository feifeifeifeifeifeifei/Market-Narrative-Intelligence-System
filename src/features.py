import numpy as np
import pandas as pd

from src.clean import html_to_text, is_blank


def add_daily_returns(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    result = df.copy()
    for ticker in tickers:
        open_col = f"{ticker}_open"
        close_col = f"{ticker}_close"
        return_col = f"{ticker}_daily_return"
        if open_col not in result.columns or close_col not in result.columns:
            continue

        open_price = pd.to_numeric(result[open_col], errors="coerce")
        close_price = pd.to_numeric(result[close_col], errors="coerce")
        valid = (open_price.notna() & close_price.notna() & (open_price != 0)).to_numpy()
        result[return_col] = np.divide(
            (close_price - open_price).to_numpy(dtype=float),
            open_price.to_numpy(dtype=float),
            out=np.full(len(result), np.nan),
            where=valid,
        )
    return result


def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    engagement_columns = ["replies_count", "reblogs_count", "favourites_count"]
    for column in engagement_columns:
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    result["total_engagement"] = result[engagement_columns].sum(axis=1)
    result["log_engagement"] = np.log1p(result["total_engagement"])
    return result


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "cleaned_text" not in result.columns:
        result["cleaned_text"] = ""
    cleaned_text = result["cleaned_text"].fillna("").astype(str)
    source_text = (
        result["text"].fillna("").astype(str)
        if "text" in result.columns
        else cleaned_text
    )
    content_html = (
        result["content_html"].fillna("").astype(str)
        if "content_html" in result.columns
        else cleaned_text
    )

    result["text_length"] = [
        original_text_length(text_value, html_value)
        for text_value, html_value in zip(source_text, content_html)
    ]
    result["has_url"] = source_text.str.contains(r"https?://|www\.", case=False, regex=True) | content_html.str.contains(
        r"https?://|www\.",
        case=False,
        regex=True,
    )
    result["has_media"] = result["has_media"].fillna(False).astype(bool) if "has_media" in result.columns else False
    return result


def original_text_length(text_value: object, content_html: object) -> int:
    if not is_blank(text_value):
        return len(str(text_value).strip())
    return len(html_to_text(content_html).strip())
