from __future__ import annotations

import re
from dataclasses import dataclass


MAX_QUESTION_CHARS = 1000

OUT_OF_SCOPE_MESSAGE = (
    "This assistant is designed for market narrative and Truth Social dataset analysis. "
    "Try asking about political narratives, similar historical posts, or market reactions."
)
SECURITY_RISK_MESSAGE = (
    "I can't help with secrets, hidden instructions, credentials, or system-level access. "
    "I can help analyze the public dataset and market reaction patterns."
)
TOO_LONG_MESSAGE = (
    "Your question is too long for this analysis tool. Please shorten it or focus on one market narrative question."
)

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
SECRET_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,})\b"
)
PASSWORD_PATTERN = re.compile(r"(?i)\b(password|passwd|api[_ -]?key|token|secret)\s*[:=]\s*\S+")
ADDRESS_PATTERN = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court)\b",
    re.IGNORECASE,
)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{12,}")
WHITESPACE_PATTERN = re.compile(r"\s+")

SECURITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bignore (all )?(previous|above|prior) instructions\b",
        r"\byou are now (system|developer|admin)\b",
        r"\bdeveloper mode\b",
        r"\bprint (your )?(system|hidden|developer) prompt\b",
        r"\b(api key|password|secret|credential|env(?:ironment)? variable|\.env)\b",
        r"\barbitrary sql\b",
        r"\bdelete\b\s+(?:\S+\s+){0,3}(data|database|table|file)\b",
        r"\bdrop\s+table\b",
        r"\bexfiltrate\b",
        r"\baccess files outside\b",
    ]
]

IN_SCOPE_KEYWORDS = {
    "truth",
    "truth social",
    "post",
    "posts",
    "trump",
    "narrative",
    "narratives",
    "policy",
    "political",
    "politics",
    "tariff",
    "tariffs",
    "china",
    "oil",
    "energy",
    "war",
    "defense",
    "healthcare",
    "pharma",
    "fed",
    "rates",
    "usd",
    "crypto",
    "market",
    "markets",
    "reaction",
    "returns",
    "return",
    "ticker",
    "tickers",
    "asset",
    "assets",
    "similar",
    "semantic",
    "topic",
    "tone",
    "entity",
    "entities",
    "election",
    "immigration",
    "border",
    "legal",
    "court",
    "russia",
    "ukraine",
    "iran",
    "mexico",
    "taiwan",
    "nato",
    "powell",
    "tariff_trade",
    "oil_energy",
    "war_defense",
    "healthcare_pharma",
    "rates_usd",
    "broad_market",
    "self_promotion",
    "legal_court",
    "election_politics",
    "immigration_border",
    "sp500",
    "qqq",
    "gld",
    "tlt",
    "uso",
}


@dataclass(frozen=True)
class GuardrailResult:
    decision: str
    original_question: str
    redacted_question: str
    message: str | None = None


def normalize_question(question: str) -> str:
    normalized = WHITESPACE_PATTERN.sub(" ", question.strip())
    normalized = REPEATED_CHAR_PATTERN.sub(lambda match: match.group(1) * 12, normalized)
    return normalized


def redact_pii(text: str) -> str:
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = ADDRESS_PATTERN.sub("[REDACTED_ADDRESS]", redacted)
    redacted = PASSWORD_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED_SECRET]", redacted)
    redacted = SECRET_PATTERN.sub("[REDACTED_SECRET]", redacted)
    redacted = CREDIT_CARD_PATTERN.sub(redact_credit_card, redacted)
    return redacted


def classify_question(question: str) -> GuardrailResult:
    raw_question = question.strip()
    normalized = normalize_question(question)

    if len(raw_question) > MAX_QUESTION_CHARS:
        return GuardrailResult("too_long", normalized, redact_pii(normalized), TOO_LONG_MESSAGE)

    redacted = redact_pii(normalized)
    if any(pattern.search(redacted) for pattern in SECURITY_PATTERNS):
        return GuardrailResult("security_risk", normalized, redacted, SECURITY_RISK_MESSAGE)
    if is_in_scope(redacted):
        return GuardrailResult("in_scope", normalized, redacted)
    return GuardrailResult("out_of_scope", normalized, redacted, OUT_OF_SCOPE_MESSAGE)


def is_in_scope(question: str) -> bool:
    lowered = question.lower()
    return any(keyword_matches(lowered, keyword) for keyword in IN_SCOPE_KEYWORDS)


def keyword_matches(text: str, keyword: str) -> bool:
    pattern = re.escape(keyword).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9_]){pattern}(?![a-z0-9_])", text) is not None


def redact_credit_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and luhn_ok(digits):
        return "[REDACTED_SECRET]"
    return match.group(0)


def luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
