from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import CLASSIFIED_EVENTS_PATH, POWERBI_EXPORT_PATH
from src.classify import write_checkpoint
from src.ticker_mapping import add_selected_tickers


POWERBI_COLUMNS = [
    "post_id",
    "datetime",
    "date",
    "time_eastern",
    "during_market_hours",
    "market_period",
    "url",
    "cleaned_text",
    "is_president",
    "has_media",
    "replies_count",
    "reblogs_count",
    "favourites_count",
    "total_engagement",
    "log_engagement",
    "text_length",
    "has_url",
    "primary_topic",
    "tone",
    "entities",
    "market_relevance",
    "policy_direction",
    "classification_reason",
    "classification_status",
    "classification_fallback",
    "classification_text_truncated",
    "selected_tickers",
    "selected_return_columns",
    "sp500_daily_return",
    "dia_daily_return",
    "qqq_daily_return",
    "djt_daily_return",
    "lmt_daily_return",
    "war_daily_return",
    "cnrg_daily_return",
    "xlv_daily_return",
    "xph_daily_return",
    "gld_daily_return",
    "uso_daily_return",
    "xli_daily_return",
    "eww_daily_return",
    "vgk_daily_return",
    "ibit_daily_return",
    "fxi_daily_return",
    "tlt_daily_return",
    "uup_daily_return",
]


def available_export_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in POWERBI_COLUMNS if column in df.columns]


def prepare_powerbi_export(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_selected_tickers(df)
    return enriched.loc[:, available_export_columns(enriched)].copy()


def write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)


def run_powerbi_export(
    input_path: Path = CLASSIFIED_EVENTS_PATH,
    output_path: Path = POWERBI_EXPORT_PATH,
    update_classified_path: Path | None = CLASSIFIED_EVENTS_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    enriched = add_selected_tickers(df)

    if update_classified_path is not None:
        write_checkpoint(enriched, update_classified_path)

    export_df = prepare_powerbi_export(enriched)
    write_csv_atomic(export_df, output_path)
    return export_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classified events for Power BI.")
    parser.add_argument("--input-path", type=Path, default=CLASSIFIED_EVENTS_PATH)
    parser.add_argument("--output-path", type=Path, default=POWERBI_EXPORT_PATH)
    parser.add_argument(
        "--no-update-classified",
        action="store_true",
        help="Write only the CSV export and leave the classified Parquet unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_path = None if args.no_update_classified else args.input_path
    export_df = run_powerbi_export(
        input_path=args.input_path,
        output_path=args.output_path,
        update_classified_path=update_path,
    )
    print(f"Wrote {len(export_df):,} rows to {args.output_path}")


if __name__ == "__main__":
    main()
