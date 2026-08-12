from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.build_chroma import get_chroma_client, get_collection
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR
from src.embeddings import EmbeddingProvider, create_embedding_provider

SEARCH_FILTER_FIELDS = {
    "primary_topic",
    "tone",
    "market_relevance",
    "policy_direction",
}


def similarity_from_distance(distance: float | None) -> float | None:
    if distance is None:
        return None
    return 1.0 - float(distance)


def search_similar_posts(
    query: str,
    top_k: int | None = 10,
    persist_dir: Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_provider_kind: str = "auto",
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")

    provider = embedding_provider or create_embedding_provider(embedding_provider_kind)
    client = get_chroma_client(persist_dir)
    collection = get_collection(client, collection_name)
    validate_embedding_provider_matches_collection(provider, collection)
    query_embedding = provider.embed_texts([query])[0]
    n_results = top_k if top_k is not None else max(1, int(collection.count()))
    where_filter = build_chroma_where(filters)

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances", "embeddings"],
        **({"where": where_filter} if where_filter else {}),
    )
    return normalize_query_results(raw_results)


def build_chroma_where(filters: dict[str, str] | None) -> dict[str, Any] | None:
    normalized = normalize_search_filters(filters)
    if not normalized:
        return None
    clauses = [{field: value} for field, value in normalized.items()]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def normalize_search_filters(filters: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field, value in (filters or {}).items():
        if field not in SEARCH_FILTER_FIELDS:
            raise ValueError(f"Unsupported search filter: {field}.")
        clean_value = str(value).strip()
        if clean_value and clean_value.lower() != "all":
            normalized[field] = clean_value
    return normalized


def validate_embedding_provider_matches_collection(
    provider: EmbeddingProvider,
    collection: Any,
) -> None:
    expected_provider = (collection.metadata or {}).get("embedding_provider")
    if expected_provider and expected_provider != provider.name:
        raise ValueError(
            "Embedding provider mismatch: collection was built with "
            f"`{expected_provider}`, but search is using `{provider.name}`."
        )


def normalize_query_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    ids = first_result_list(raw_results, "ids")
    documents = first_result_list(raw_results, "documents")
    metadatas = first_result_list(raw_results, "metadatas")
    distances = first_result_list(raw_results, "distances")
    embeddings = first_embedding_list(raw_results, "embeddings")

    normalized: list[dict[str, Any]] = []
    for index, post_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        embedding = embeddings[index] if index < len(embeddings) else None
        normalized.append(
            {
                "post_id": post_id,
                "score": similarity_from_distance(distance),
                "distance": distance,
                "cleaned_text": documents[index] if index < len(documents) else "",
                "metadata": metadata,
                "embedding": embedding,
            }
        )
    return normalized


def first_result_list(raw_results: dict[str, Any], key: str) -> list[Any]:
    values = raw_results.get(key) or [[]]
    if not values:
        return []
    return values[0] or []


def to_float_list(embedding: Any) -> list[float] | None:
    if embedding is None:
        return None
    try:
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(x) for x in embedding]
    except (TypeError, ValueError):
        return None


def first_embedding_list(raw_results: dict[str, Any], key: str) -> list[list[float] | None]:
    values = raw_results.get(key)
    if values is None or len(values) == 0:
        return []
    embeddings = values[0]
    if embeddings is None or (isinstance(embeddings, list) and len(embeddings) == 0):
        return []
    return [to_float_list(emb) for emb in embeddings]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search similar posts in ChromaDB.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DB_DIR)
    parser.add_argument("--collection-name", default=CHROMA_COLLECTION_NAME)
    parser.add_argument("--embedding-provider", choices=["auto", "openai", "hashing", "local_hashing"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = search_similar_posts(
        query=args.query,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_provider_kind=args.embedding_provider,
    )
    for result in results:
        score = result["score"]
        score_text = "" if score is None else f"{score:.4f}"
        print(f"{result['post_id']}\t{score_text}\t{result['cleaned_text'][:160]}")


if __name__ == "__main__":
    main()
