from fastapi.testclient import TestClient

import app.main as api


def test_health_endpoint() -> None:
    client = TestClient(api.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}
    assert "events_available" in response.json()
    assert "chroma_available" in response.json()


def test_topics_endpoint() -> None:
    client = TestClient(api.app)

    response = client.get("/api/topics")

    assert response.status_code == 200
    assert response.json()["tariff_trade"] == ["SP500", "QQQ", "FXI", "UUP", "TLT"]


def test_filter_options_endpoint() -> None:
    client = TestClient(api.app)

    response = client.get("/api/filter-options")

    assert response.status_code == 200
    assert "primary_topic" not in response.json()
    assert "aggressive" in response.json()["tone"]


def test_analyze_refuses_out_of_scope_question() -> None:
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"question": "Write me a love letter", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["query_type"] == "refusal"


def test_analyze_uses_guarded_question_and_returns_api_shape(monkeypatch) -> None:
    def fake_analyze_similar_events(**kwargs):
        assert kwargs["query"] == "China tariff market reaction"
        assert kwargs["top_k"] is None
        assert kwargs["filters"] == {"tone": "threatening"}
        return {
            "summary": "Retrieved 1 similar post.",
            "query_type": "similar_event_analysis",
            "filters": {"tone": "threatening"},
            "selected_topics": [{"primary_topic": "tariff_trade", "count": 1}],
            "selected_tickers": ["SP500", "QQQ"],
            "similar_posts": [
                {
                    "post_id": "1",
                    "date": "2026-01-01T00:00:00",
                    "cleaned_text": "China tariff market reaction",
                    "similarity_score": 0.9,
                    "primary_topic": "tariff_trade",
                    "tone": "threatening",
                    "market_relevance": "high",
                    "policy_direction": "escalation",
                }
            ],
            "market_reaction": [{"ticker": "QQQ", "avg_daily_return": 0.01, "median_daily_return": 0.01, "sample_size": 1}],
            "retrieved_count": 1,
        }

    monkeypatch.setattr(api, "analyze_similar_events", fake_analyze_similar_events)
    client = TestClient(api.app)

    response = client.post(
        "/api/analyze",
        json={
            "question": "China tariff market reaction",
            "tone": "threatening",
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["query_type"] == "similar_event_analysis"
    assert body["selected_topic"] == "tariff_trade"
    assert body["selected_topics"] == [{"primary_topic": "tariff_trade", "count": 1}]
    assert body["filters"] == {"tone": "threatening"}
    assert body["similar_posts"][0]["post_id"] == "1"
    assert body["similar_posts"][0]["market_relevance"] == "high"


def test_analyze_returns_friendly_value_error(monkeypatch) -> None:
    def fake_analyze_similar_events(**kwargs):
        raise ValueError("Embedding provider mismatch: collection was built with `openai`.")

    monkeypatch.setattr(api, "analyze_similar_events", fake_analyze_similar_events)
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"question": "China tariff market reaction", "top_k": 1})

    assert response.status_code == 400
    assert "Embedding provider mismatch" in response.json()["detail"]


def test_analyze_returns_friendly_runtime_error(monkeypatch) -> None:
    def fake_analyze_similar_events(**kwargs):
        raise RuntimeError("duckdb is not installed")

    monkeypatch.setattr(api, "analyze_similar_events", fake_analyze_similar_events)
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"question": "China tariff market reaction", "top_k": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Analysis backend is not ready. Check ChromaDB, DuckDB, and data files."
