from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PrimaryTopic(str, Enum):
    TARIFF_TRADE = "tariff_trade"
    OIL_ENERGY = "oil_energy"
    WAR_DEFENSE = "war_defense"
    HEALTHCARE_PHARMA = "healthcare_pharma"
    RATES_USD = "rates_usd"
    CRYPTO = "crypto"
    BROAD_MARKET = "broad_market"
    IMMIGRATION_BORDER = "immigration_border"
    LEGAL_COURT = "legal_court"
    ELECTION_POLITICS = "election_politics"
    SELF_PROMOTION = "self_promotion"
    OTHER = "other"


class Tone(str, Enum):
    AGGRESSIVE = "aggressive"
    DEESCALATING = "deescalating"
    PROMOTIONAL = "promotional"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"
    THREATENING = "threatening"
    PRAISING = "praising"


class MarketRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PolicyDirection(str, Enum):
    ESCALATION = "escalation"
    DEESCALATION = "deescalation"
    EASING = "easing"
    UNCERTAINTY = "uncertainty"
    NEUTRAL = "neutral"


class NarrativeClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_topic: PrimaryTopic
    tone: Tone
    entities: list[str] = Field(default_factory=list)
    market_relevance: MarketRelevance
    policy_direction: PolicyDirection
    reason: str = Field(min_length=1, max_length=220)

    @field_validator("entities")
    @classmethod
    def clean_entities(cls, entities: list[str]) -> list[str]:
        cleaned: list[str] = []
        for entity in entities:
            normalized = str(entity).strip()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized[:80])
        return cleaned[:12]

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, reason: str) -> str:
        return " ".join(reason.split())


def fallback_classification() -> NarrativeClassification:
    return NarrativeClassification(
        primary_topic=PrimaryTopic.OTHER,
        tone=Tone.NEUTRAL,
        entities=[],
        market_relevance=MarketRelevance.LOW,
        policy_direction=PolicyDirection.NEUTRAL,
        reason="Fallback classification used because no valid model output was available.",
    )
