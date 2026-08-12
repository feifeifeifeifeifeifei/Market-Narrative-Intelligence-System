from __future__ import annotations

from collections import deque

import numpy as np

from src.config import (
    DBSCAN_EPS_CEIL,
    DBSCAN_EPS_FLOOR,
    DBSCAN_EPS_QUANTILE,
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
