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


from src.clustering import ClusterOutcome, cluster_search_results


def _result(post_id, embedding, topic, score):
    return {
        "post_id": post_id,
        "embedding": embedding,
        "score": score,
        "cleaned_text": f"text for {post_id}",
        "metadata": {"primary_topic": topic},
    }


def _two_cluster_results():
    return [
        _result("a", [1.0, 0.0], "tariff_trade", 0.90),
        _result("b", [0.99, 0.02], "tariff_trade", 0.80),
        _result("c", [0.0, 1.0], "oil_energy", 0.85),
        _result("d", [0.02, 0.99], "oil_energy", 0.70),
        _result("e", [-1.0, 0.0], "other", 0.30),
    ]


def test_cluster_search_results_groups_and_flags_noise():
    outcome = cluster_search_results(_two_cluster_results(), eps=0.1, min_samples=2)
    assert isinstance(outcome, ClusterOutcome)
    assert outcome.clustering_applied is True
    assert outcome.noise_count == 1
    assert {label for label in outcome.labels if label != NOISE_LABEL} == {0, 1}
    assert outcome.labels[4] == NOISE_LABEL
    assert len(outcome.narratives) == 2


def test_cluster_search_results_narratives_sorted_by_size_then_similarity():
    outcome = cluster_search_results(_two_cluster_results(), eps=0.1, min_samples=2)
    first = outcome.narratives[0]
    assert first["cluster_id"] == 0
    assert first["size"] == 2
    assert first["dominant_topic"] == "tariff_trade"
    assert first["representative_post_id"] == "a"
    assert first["avg_similarity"] == pytest.approx(0.85)
    assert first["post_ids"] == ["a", "b"]


def test_cluster_search_results_noop_without_embeddings():
    results = [
        {"post_id": "a", "score": 0.9, "cleaned_text": "x", "metadata": {}},
        {"post_id": "b", "score": 0.8, "cleaned_text": "y", "metadata": {}},
    ]
    outcome = cluster_search_results(results)
    assert outcome.clustering_applied is False
    assert outcome.labels == [None, None]
    assert outcome.narratives == []
    assert outcome.noise_count == 0


def test_cluster_search_results_noop_when_too_few_points():
    outcome = cluster_search_results([_result("a", [1.0, 0.0], "t", 0.9)], min_samples=2)
    assert outcome.clustering_applied is False
    assert outcome.labels == [None]


def test_cluster_search_results_is_deterministic():
    results = _two_cluster_results()
    first = cluster_search_results(results, eps=0.1, min_samples=2)
    second = cluster_search_results(results, eps=0.1, min_samples=2)
    assert first.labels == second.labels


def test_cluster_search_results_truncates_representative_text():
    long_text = "x" * 500
    results = [
        {"post_id": "a", "embedding": [1.0, 0.0], "score": 0.9, "cleaned_text": long_text, "metadata": {"primary_topic": "t"}},
        {"post_id": "b", "embedding": [0.99, 0.02], "score": 0.8, "cleaned_text": long_text, "metadata": {"primary_topic": "t"}},
    ]
    outcome = cluster_search_results(results, eps=0.1, min_samples=2)
    assert len(outcome.narratives[0]["representative_text"]) <= 241  # 240 + ellipsis
