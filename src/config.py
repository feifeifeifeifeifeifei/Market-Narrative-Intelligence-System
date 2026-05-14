from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "trump_truth_social.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_EVENTS_PATH = PROCESSED_DIR / "cleaned_events.parquet"
CLASSIFIED_EVENTS_PATH = PROCESSED_DIR / "classified_events.parquet"
POWERBI_EXPORT_PATH = PROCESSED_DIR / "powerbi_export.csv"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "market_narrative_posts"

DEFAULT_CLASSIFICATION_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

CORE_COLUMNS = [
    "post_id",
    "datetime",
    "date",
    "time",
    "day_of_week",
    "text",
    "content_html",
    "url",
    "is_president",
    "is_president_elect",
    "replies_count",
    "reblogs_count",
    "favourites_count",
    "has_media",
    "media_urls",
    "links",
    "image_alt_text",
    "time_eastern",
    "during_market_hours",
    "market_period",
]

CATEGORY_COLUMNS = [
    "cat_attacking_individual",
    "cat_attacking_opposition",
    "cat_deescalating",
    "cat_enacting_aggressive",
    "cat_enacting_nonaggressive",
    "cat_other",
    "cat_praising_endorsing",
    "cat_self_promotion",
    "cat_threatening_intl",
    "gdelt_military",
    "gdelt_sanctions",
    "gdelt_threat",
    "gdelt_protest",
    "gdelt_force_posture",
    "gdelt_diplomatic",
    "gdelt_material_conflict",
    "gdelt_verbal_conflict",
    "gdelt_material_cooperation",
    "gdelt_verbal_cooperation",
    "gdelt_goldstein_avg",
    "gdelt_avg_tone",
    "gdelt_total_events",
    "gdelt_military_pct",
    "gdelt_sanctions_pct",
    "gdelt_threat_pct",
    "gdelt_protest_pct",
    "gdelt_force_posture_pct",
    "gdelt_diplomatic_pct",
    "gdelt_military_zscore",
    "gdelt_sanctions_zscore",
    "gdelt_threat_zscore",
    "gdelt_protest_zscore",
    "gdelt_material_conflict_zscore",
    "gdelt_military_delta",
    "gdelt_sanctions_delta",
    "gdelt_threat_delta",
    "gdelt_protest_delta",
    "gdelt_material_conflict_delta",
    "gdelt_goldstein_avg_delta",
    "gdelt_avg_tone_delta",
]

TICKERS = [
    "sp500",
    "dia",
    "qqq",
    "djt",
    "lmt",
    "war",
    "cnrg",
    "xlv",
    "xph",
    "gld",
    "uso",
    "xli",
    "eww",
    "vgk",
    "ibit",
    "fxi",
    "tlt",
    "uup",
]
