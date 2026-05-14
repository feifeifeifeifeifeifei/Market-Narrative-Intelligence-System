import pandas as pd

from src.schemas import PrimaryTopic
from src.ticker_mapping import (
    TOPIC_TO_TICKERS,
    add_selected_tickers,
    return_column_for_ticker,
    selected_tickers_for_topic,
)


def test_topic_mapping_matches_required_tariff_trade_tickers() -> None:
    assert TOPIC_TO_TICKERS["tariff_trade"] == ["SP500", "QQQ", "FXI", "UUP", "TLT"]


def test_every_primary_topic_has_ticker_mapping() -> None:
    assert {topic.value for topic in PrimaryTopic} == set(TOPIC_TO_TICKERS)


def test_unknown_topic_uses_other_mapping() -> None:
    assert selected_tickers_for_topic("unexpected") == ["SP500", "QQQ"]


def test_missing_topic_uses_empty_mapping() -> None:
    assert selected_tickers_for_topic(None) == []
    assert selected_tickers_for_topic(float("nan")) == []


def test_return_column_for_ticker_uses_daily_return_naming() -> None:
    assert return_column_for_ticker("SP500") == "sp500_daily_return"


def test_add_selected_tickers_adds_serialized_ticker_and_return_columns() -> None:
    df = pd.DataFrame({"primary_topic": ["war_defense", "other", None]})

    result = add_selected_tickers(df)

    assert result.loc[0, "selected_tickers"] == "LMT,WAR,GLD"
    assert result.loc[0, "selected_return_columns"] == "lmt_daily_return,war_daily_return,gld_daily_return"
    assert result.loc[1, "selected_tickers"] == "SP500,QQQ"
    assert result.loc[2, "selected_tickers"] == ""
