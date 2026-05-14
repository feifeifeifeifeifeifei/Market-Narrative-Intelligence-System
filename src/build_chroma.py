from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    CLASSIFIED_EVENTS_PATH,
)
from src.embeddings import EmbeddingProvider, create_embedding_provider


CHROMA_METADATA_COLUMNS = [
    "date",
    "datetime",
    "primary_topic",
    "tone",
    "entities",
    "market_relevance",
    "policy_direction",
    "is_president",
]


def require_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return chromadb


def get_chroma_client(persist_dir: Path = CHROMA_DB_DIR) -> Any:
    chromadb = require_chromadb()
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def reset_collection(
    client: Any,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_provider_name: str = "unknown",
) -> Any:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": embedding_provider_name,
        },
    )


def get_collection(
    client: Any,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Any:
    return client.get_collection(name=collection_name)


def metadata_value(value: Any) -> str | bool | int | float:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bool | int | float):
        return value
    return str(value)


def row_metadata(row: pd.Series) -> dict[str, str | bool | int | float]:
    metadata: dict[str, str | bool | int | float] = {}
    for column in CHROMA_METADATA_COLUMNS:
        if column in row.index:
            metadata[column] = metadata_value(row[column])
    return metadata


def chroma_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"post_id", "cleaned_text"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Input data is missing required columns: {missing_text}")

    rows = df.copy()
    rows["post_id"] = rows["post_id"].astype(str)
    rows["cleaned_text"] = rows["cleaned_text"].fillna("").astype(str).str.strip()
    rows = rows.loc[rows["cleaned_text"] != ""].copy()
    rows = rows.drop_duplicates("post_id", keep="last")
    return rows


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def upsert_dataframe(
    collection: Any,
    df: pd.DataFrame,
    embedding_provider: EmbeddingProvider,
    batch_size: int = 500,
    show_progress: bool = False,
    skip_existing: bool = False,
) -> int:
    rows = chroma_rows(df)
    if rows.empty:
        return 0
    if skip_existing:
        rows = rows.loc[~rows["post_id"].isin(existing_collection_ids(collection))].copy()
        if rows.empty:
            return 0

    ids = rows["post_id"].tolist()
    documents = rows["cleaned_text"].tolist()
    metadatas = [row_metadata(row) for _, row in rows.iterrows()]

    batches = list(zip(
        batched(ids, batch_size),
        batched(documents, batch_size),
        batched(metadatas, batch_size),
    ))
    written = 0
    for id_batch, doc_batch, metadata_batch in iter_with_progress(
        batches,
        show_progress=show_progress,
        desc="upsert_chroma",
    ):
        embeddings = embedding_provider.embed_texts(doc_batch)
        collection.upsert(
            ids=id_batch,
            documents=doc_batch,
            metadatas=metadata_batch,
            embeddings=embeddings,
        )
        written += len(id_batch)
    return written


def existing_collection_ids(collection: Any, batch_size: int = 5000) -> set[str]:
    try:
        total = collection.count()
    except Exception:
        return set()

    ids: set[str] = set()
    for offset in range(0, total, batch_size):
        page = collection.get(limit=batch_size, offset=offset, include=[])
        ids.update(str(post_id) for post_id in page.get("ids", []))
    return ids


def iter_with_progress(items: list[Any], show_progress: bool, desc: str) -> Any:
    if not show_progress:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, total=len(items))


def build_chroma_collection(
    input_path: Path = CLASSIFIED_EVENTS_PATH,
    persist_dir: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_provider_kind: str = "auto",
    batch_size: int = 500,
    reset: bool = True,
    show_progress: bool = True,
    resume: bool = False,
) -> int:
    df = pd.read_parquet(input_path)
    provider = embedding_provider or create_embedding_provider(embedding_provider_kind)
    client = get_chroma_client(persist_dir)
    collection = (
        reset_collection(client, collection_name, provider.name)
        if reset
        else client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_provider": provider.name,
            },
        )
    )
    return upsert_dataframe(
        collection,
        df,
        provider,
        batch_size=batch_size,
        show_progress=show_progress,
        skip_existing=resume and not reset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB collection for post retrieval.")
    parser.add_argument("--input-path", type=Path, default=CLASSIFIED_EVENTS_PATH)
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DB_DIR)
    parser.add_argument("--collection-name", default=CHROMA_COLLECTION_NAME)
    parser.add_argument("--embedding-provider", choices=["auto", "openai", "hashing", "local_hashing"], default="auto")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_chroma_collection(
        input_path=args.input_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_provider_kind=args.embedding_provider,
        batch_size=args.batch_size,
        reset=not args.no_reset,
        show_progress=not args.no_progress,
        resume=args.resume,
    )
    print(f"Wrote {count:,} documents to Chroma collection `{args.collection_name}`")


if __name__ == "__main__":
    main()
