import pandas as pd

from src.export_powerbi import prepare_powerbi_export, run_powerbi_export


def test_prepare_powerbi_export_includes_selected_tickers_and_core_fields() -> None:
    df = pd.DataFrame(
        {
            "post_id": ["1"],
            "date": pd.to_datetime(["2026-01-01"]),
            "time_eastern": ["09:45:00"],
            "during_market_hours": [True],
            "market_period": ["market_hours"],
            "cleaned_text": ["China tariff threat"],
            "primary_topic": ["tariff_trade"],
            "tone": ["threatening"],
            "market_relevance": ["high"],
            "policy_direction": ["escalation"],
            "qqq_daily_return": [0.01],
            "fxi_daily_return": [-0.02],
        }
    )

    result = prepare_powerbi_export(df)

    assert result.loc[0, "selected_tickers"] == "SP500,QQQ,FXI,UUP,TLT"
    assert result.loc[0, "selected_return_columns"] == (
        "sp500_daily_return,qqq_daily_return,fxi_daily_return,uup_daily_return,tlt_daily_return"
    )
    assert "post_id" in result.columns
    assert "cleaned_text" in result.columns
    assert "during_market_hours" in result.columns
    assert "policy_direction" in result.columns


def test_run_powerbi_export_writes_csv_and_updates_classified_parquet(tmp_path) -> None:
    input_path = tmp_path / "classified.parquet"
    output_path = tmp_path / "powerbi_export.csv"
    df = pd.DataFrame(
        {
            "post_id": ["1"],
            "date": pd.to_datetime(["2026-01-01"]),
            "cleaned_text": ["Defense spending"],
            "primary_topic": ["war_defense"],
            "tone": ["neutral"],
            "market_relevance": ["medium"],
            "policy_direction": ["neutral"],
            "during_market_hours": [False],
        }
    )
    df.to_parquet(input_path, index=False)

    export_df = run_powerbi_export(
        input_path=input_path,
        output_path=output_path,
        update_classified_path=input_path,
    )
    updated = pd.read_parquet(input_path)
    csv_df = pd.read_csv(output_path)

    assert output_path.exists()
    assert export_df.loc[0, "selected_tickers"] == "LMT,WAR,GLD"
    assert updated.loc[0, "selected_tickers"] == "LMT,WAR,GLD"
    assert csv_df.loc[0, "selected_tickers"] == "LMT,WAR,GLD"


def test_run_powerbi_export_can_skip_classified_parquet_update(tmp_path) -> None:
    input_path = tmp_path / "classified.parquet"
    output_path = tmp_path / "powerbi_export.csv"
    df = pd.DataFrame(
        {
            "post_id": ["1"],
            "cleaned_text": ["Unchanged"],
            "primary_topic": ["other"],
        }
    )
    df.to_parquet(input_path, index=False)

    run_powerbi_export(
        input_path=input_path,
        output_path=output_path,
        update_classified_path=None,
    )
    updated = pd.read_parquet(input_path)

    assert "selected_tickers" not in updated.columns
    assert output_path.exists()
