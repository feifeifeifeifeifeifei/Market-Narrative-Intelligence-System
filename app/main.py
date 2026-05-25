from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.analytics import analyze_similar_events
from src.build_chroma import get_chroma_client, get_collection
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, CLASSIFIED_EVENTS_PATH
from src.guardrails import classify_question
from src.schemas import MarketRelevance, PolicyDirection, Tone
from src.ticker_mapping import TOPIC_TO_TICKERS


MAX_API_TOP_K = 50
API_EMBEDDING_PROVIDER = os.getenv("API_EMBEDDING_PROVIDER", "openai")
logger = logging.getLogger(__name__)
CHROMA_LOCK = threading.Lock()


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=MAX_API_TOP_K)
    tone: Tone | None = None
    market_relevance: MarketRelevance | None = None
    policy_direction: PolicyDirection | None = None


class HealthResponse(BaseModel):
    status: str
    collection: str
    events_available: bool
    chroma_available: bool
    chroma_count: int | None = None
    embedding_provider: str


class SimilarPostResponse(BaseModel):
    post_id: Any = None
    date: Any = None
    cleaned_text: str | None = None
    similarity_score: float | None = None
    primary_topic: str | None = None
    tone: str | None = None
    market_relevance: str | None = None
    policy_direction: str | None = None


class AnalyzeResponse(BaseModel):
    summary: str | None
    query_type: str
    guardrail_decision: str
    redacted_question: str
    selected_topic: str | None
    selected_topics: list[dict[str, Any]] = Field(default_factory=list)
    selected_tickers: list[str]
    filters: dict[str, str] = Field(default_factory=dict)
    similar_posts: list[SimilarPostResponse]
    market_reaction: list[dict[str, Any]]
    retrieved_count: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Market Narrative Intelligence API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        events_available = CLASSIFIED_EVENTS_PATH.exists()
        chroma_available = False
        chroma_count: int | None = None
        try:
            with CHROMA_LOCK:
                collection = get_collection(get_chroma_client(CHROMA_DB_DIR), CHROMA_COLLECTION_NAME)
                chroma_count = int(collection.count())
            chroma_available = True
        except Exception:
            logger.info("Chroma collection health probe failed", exc_info=True)

        return HealthResponse(
            status="ok" if events_available and chroma_available else "degraded",
            collection=CHROMA_COLLECTION_NAME,
            events_available=events_available,
            chroma_available=chroma_available,
            chroma_count=chroma_count,
            embedding_provider=API_EMBEDDING_PROVIDER,
        )

    @app.get("/api/topics")
    def topics() -> dict[str, list[str]]:
        return TOPIC_TO_TICKERS

    @app.get("/api/filter-options")
    def filter_options() -> dict[str, list[str]]:
        return {
            "tone": [item.value for item in Tone],
            "market_relevance": [item.value for item in MarketRelevance],
            "policy_direction": [item.value for item in PolicyDirection],
        }

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        guardrail = classify_question(request.question)
        logger.info(
            "analyze decision=%s redacted_question=%s top_k=%s filters=%s",
            guardrail.decision,
            guardrail.redacted_question,
            request.top_k,
            request_filters(request),
        )
        if guardrail.decision != "in_scope":
            return AnalyzeResponse(
                summary=guardrail.message,
                query_type="refusal",
                guardrail_decision=guardrail.decision,
                redacted_question=guardrail.redacted_question,
                selected_topic=None,
                selected_topics=[],
                selected_tickers=[],
                filters=request_filters(request),
                similar_posts=[],
                market_reaction=[],
            )

        try:
            with CHROMA_LOCK:
                result = analyze_similar_events(
                    query=guardrail.redacted_question,
                    top_k=request.top_k,
                    filters=request_filters(request),
                    embedding_provider_kind=API_EMBEDDING_PROVIDER,
                )
        except FileNotFoundError as exc:
            logger.exception("Classified events parquet is unavailable")
            raise HTTPException(
                status_code=503,
                detail="Classified events parquet is not available. Run the ETL and classification pipeline.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("Analysis backend is not ready")
            raise HTTPException(
                status_code=503,
                detail="Analysis backend is not ready. Check ChromaDB, DuckDB, and data files.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected analyze failure")
            raise HTTPException(
                status_code=500,
                detail="Unexpected backend error. See server logs.",
            ) from exc

        # This is the most common retrieved topic, not a separate classifier for the user's question.
        selected_topic = (
            result["selected_topics"][0]["primary_topic"]
            if result.get("selected_topics")
            else None
        )
        return AnalyzeResponse(
            summary=result["summary"],
            query_type=result["query_type"],
            guardrail_decision=guardrail.decision,
            redacted_question=guardrail.redacted_question,
            selected_topic=selected_topic,
            selected_topics=result.get("selected_topics", []),
            selected_tickers=result["selected_tickers"],
            filters=result.get("filters", request_filters(request)),
            similar_posts=slim_similar_posts(result["similar_posts"]),
            market_reaction=result["market_reaction"],
            retrieved_count=result["retrieved_count"],
        )

    return app


def request_filters(request: AnalyzeRequest) -> dict[str, str]:
    values = {
        "tone": request.tone,
        "market_relevance": request.market_relevance,
        "policy_direction": request.policy_direction,
    }
    return {field: value.value for field, value in values.items() if value is not None}


def slim_similar_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "post_id": post.get("post_id"),
            "date": post.get("date"),
            "cleaned_text": post.get("cleaned_text"),
            "similarity_score": post.get("similarity_score"),
            "primary_topic": post.get("primary_topic"),
            "tone": post.get("tone"),
            "market_relevance": post.get("market_relevance"),
            "policy_direction": post.get("policy_direction"),
        }
        for post in posts
    ]


app = create_app()
