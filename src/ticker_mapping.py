from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


TOPIC_TO_TICKERS: dict[str, list[str]] = {
    "tariff_trade": ["SP500", "QQQ", "FXI", "UUP", "TLT"],
    "oil_energy": ["USO", "CNRG", "XLI"],
    "war_defense": ["LMT", "WAR", "GLD"],
    "healthcare_pharma": ["XLV", "XPH"],
    "rates_usd": ["TLT", "UUP", "GLD", "QQQ"],
    "crypto": ["IBIT", "QQQ"],
    "broad_market": ["SP500", "DIA", "QQQ"],
    "self_promotion": ["DJT", "SP500"],
    "legal_court": ["SP500", "DIA", "QQQ"],
    "election_politics": ["SP500", "DIA", "QQQ", "TLT", "UUP"],
    "immigration_border": ["EWW", "SP500", "DIA"],
    "other": ["SP500", "QQQ"],
}


def selected_tickers_for_topic(primary_topic: Any) -> list[str]:
    if pd.isna(primary_topic):
        return []
    topic = str(primary_topic).strip()
    if not topic:
        return []
    return TOPIC_TO_TICKERS.get(topic, TOPIC_TO_TICKERS["other"]).copy()


def serialize_tickers(tickers: Iterable[str]) -> str:
    return ",".join(tickers)


def return_column_for_ticker(ticker: str) -> str:
    return f"{ticker.lower()}_daily_return"


def selected_return_columns(tickers: Iterable[str]) -> list[str]:
    return [return_column_for_ticker(ticker) for ticker in tickers]


def add_selected_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_topic" not in df.columns:
        raise ValueError("Input data must include a primary_topic column.")

    result = df.copy()
    selected_ticker_lists = result["primary_topic"].map(selected_tickers_for_topic)
    result["selected_tickers"] = selected_ticker_lists.map(serialize_tickers)
    result["selected_return_columns"] = selected_ticker_lists.map(
        lambda ticker_list: serialize_tickers(selected_return_columns(ticker_list))
    )
    return result
