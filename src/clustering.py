from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import (
    CLUSTER_REP_TEXT_MAXLEN,
    DBSCAN_EPS_CEIL,
    DBSCAN_EPS_FLOOR,
    DBSCAN_EPS_QUANTILE,
    DBSCAN_MIN_SAMPLES,
)

NOISE_LABEL = -1


def cosine_distance_matrix(embeddings: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    distances = 1.0 - unit @ unit.T
    return np.clip(distances, 0.0, 2.0)


def adaptive_eps(
    distances: np.ndarray,
    min_samples: int,
    quantile: float = DBSCAN_EPS_QUANTILE,
    floor: float = DBSCAN_EPS_FLOOR,
    ceil: float = DBSCAN_EPS_CEIL,
) -> float:
    n = distances.shape[0]
    k = min(max(min_samples - 1, 1), n - 1)
    kth_distances = np.sort(distances, axis=1)[:, k]
    eps = float(np.quantile(kth_distances, quantile))
    return float(min(max(eps, floor), ceil))


def dbscan_labels(distances: np.ndarray, eps: float, min_samples: int) -> list[int]:
    n = distances.shape[0]
    neighbors_of = [np.where(distances[i] <= eps)[0].tolist() for i in range(n)]
    labels: list[int | None] = [None] * n
    visited = [False] * n
    cluster_id = -1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors_of[i]) < min_samples:
            labels[i] = NOISE_LABEL
            continue
        cluster_id += 1
        labels[i] = cluster_id
        queue = deque(neighbors_of[i])
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                if len(neighbors_of[j]) >= min_samples:
                    queue.extend(neighbors_of[j])
            if labels[j] is None or labels[j] == NOISE_LABEL:
                labels[j] = cluster_id

    return [NOISE_LABEL if label is None else label for label in labels]


@dataclass
class ClusterOutcome:
    labels: list[int | None] = field(default_factory=list)
    narratives: list[dict[str, Any]] = field(default_factory=list)
    noise_count: int = 0
    clustering_applied: bool = False


def cluster_search_results(
    results: list[dict[str, Any]],
    *,
    eps: float | None = None,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    eps_quantile: float = DBSCAN_EPS_QUANTILE,
) -> ClusterOutcome:
    n = len(results)
    if not _can_cluster(results, min_samples):
        return ClusterOutcome(labels=[None] * n)

    embeddings = [list(result["embedding"]) for result in results]
    distances = cosine_distance_matrix(embeddings)
    resolved_eps = eps if eps is not None else adaptive_eps(distances, min_samples, eps_quantile)
    raw_labels = dbscan_labels(distances, resolved_eps, min_samples)
    labels = _renumber_by_size(raw_labels, results)
    narratives = _build_narratives(labels, results)
    noise_count = sum(1 for label in labels if label == NOISE_LABEL)
    return ClusterOutcome(
        labels=labels,
        narratives=narratives,
        noise_count=noise_count,
        clustering_applied=True,
    )


def _can_cluster(results: list[dict[str, Any]], min_samples: int) -> bool:
    if len(results) < min_samples:
        return False
    return all(_has_embedding(result.get("embedding")) for result in results)


def _has_embedding(embedding: Any) -> bool:
    if embedding is None:
        return False
    try:
        return len(embedding) > 0
    except TypeError:
        return False


def _renumber_by_size(raw_labels: list[int], results: list[dict[str, Any]]) -> list[int]:
    sizes = Counter(label for label in raw_labels if label != NOISE_LABEL)

    def sort_key(label: int) -> tuple[int, float]:
        members = [results[i] for i, value in enumerate(raw_labels) if value == label]
        avg = _avg_similarity(members)
        return (-sizes[label], -(avg if avg is not None else -1.0))

    ordered = sorted(sizes.keys(), key=sort_key)
    remap = {old: new for new, old in enumerate(ordered)}
    return [NOISE_LABEL if label == NOISE_LABEL else remap[label] for label in raw_labels]


def _build_narratives(labels: list[int], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label == NOISE_LABEL:
            continue
        clusters.setdefault(label, []).append(index)

    narratives: list[dict[str, Any]] = []
    for label in sorted(clusters.keys()):
        members = [results[i] for i in clusters[label]]
        representative = max(members, key=lambda result: _score_or(result, -1.0))
        narratives.append(
            {
                "cluster_id": label,
                "size": len(members),
                "dominant_topic": _dominant_topic(members),
                "avg_similarity": _avg_similarity(members),
                "representative_post_id": str(representative.get("post_id")),
                "representative_text": _truncate(_text_of(representative), CLUSTER_REP_TEXT_MAXLEN),
                "post_ids": [str(result.get("post_id")) for result in members],
            }
        )
    return narratives


def _dominant_topic(members: list[dict[str, Any]]) -> str:
    topics = [
        str((member.get("metadata") or {}).get("primary_topic", "")).strip()
        for member in members
    ]
    topics = [topic for topic in topics if topic]
    if not topics:
        return ""
    return Counter(topics).most_common(1)[0][0]


def _avg_similarity(members: list[dict[str, Any]]) -> float | None:
    scores = [member.get("score") for member in members if member.get("score") is not None]
    if not scores:
        return None
    return float(sum(float(score) for score in scores) / len(scores))


def _score_or(result: dict[str, Any], default: float) -> float:
    score = result.get("score")
    return float(score) if score is not None else default


def _text_of(result: dict[str, Any]) -> str:
    return str(result.get("cleaned_text") or "")


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "…"
