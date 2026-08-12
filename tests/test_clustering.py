import numpy as np
import pytest

from src.clustering import (
    NOISE_LABEL,
    adaptive_eps,
    cosine_distance_matrix,
    dbscan_labels,
)


def test_cosine_distance_matrix_identical_and_orthogonal():
    D = cosine_distance_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert D[0, 2] == pytest.approx(1.0, abs=1e-9)
    assert D[2, 2] == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_matrix_handles_zero_vector():
    D = cosine_distance_matrix([[0.0, 0.0], [1.0, 0.0]])
    # zero vector must not produce NaN
    assert not np.isnan(D).any()


def test_dbscan_two_clusters_and_one_noise():
    big = 0.9
    D = np.full((5, 5), big)
    np.fill_diagonal(D, 0.0)
    D[0, 1] = D[1, 0] = 0.05
    D[2, 3] = D[3, 2] = 0.05
    labels = dbscan_labels(D, eps=0.1, min_samples=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] == NOISE_LABEL


def test_dbscan_all_noise_when_eps_too_small():
    D = np.full((3, 3), 0.5)
    np.fill_diagonal(D, 0.0)
    labels = dbscan_labels(D, eps=0.01, min_samples=2)
    assert labels == [NOISE_LABEL, NOISE_LABEL, NOISE_LABEL]


def test_adaptive_eps_reflects_knn_distance():
    D = np.array(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.8],
            [0.8, 0.8, 0.0],
        ]
    )
    # k=1 nearest-other distances: [0.1, 0.1, 0.8]; median = 0.1
    eps = adaptive_eps(D, min_samples=2, quantile=0.5, floor=0.05, ceil=0.8)
    assert eps == pytest.approx(0.1, abs=1e-9)


def test_adaptive_eps_respects_floor_and_ceil():
    assert adaptive_eps(np.zeros((3, 3)), min_samples=2, quantile=0.5, floor=0.05, ceil=0.8) == 0.05
    big = np.full((3, 3), 2.0)
    np.fill_diagonal(big, 0.0)
    assert adaptive_eps(big, min_samples=2, quantile=0.5, floor=0.05, ceil=0.8) == 0.8
