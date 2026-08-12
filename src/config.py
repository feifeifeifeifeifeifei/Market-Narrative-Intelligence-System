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
]

CATEGORY_COLUMNS: list[str] = []

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

# DBSCAN narrative clustering (query-time, over retrieved results)
DBSCAN_MIN_SAMPLES = 2
DBSCAN_EPS_QUANTILE = 0.6
DBSCAN_EPS_FLOOR = 0.05
DBSCAN_EPS_CEIL = 0.80
CLUSTER_REP_TEXT_MAXLEN = 240
