from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.config import (
    CLASSIFIED_EVENTS_PATH,
    CLEANED_EVENTS_PATH,
    DEFAULT_CLASSIFICATION_MODEL,
)
from src.schemas import NarrativeClassification, fallback_classification


LLMCallable = Callable[[list[dict[str, str]]], str]
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CLASSIFICATION_COLUMNS = [
    "primary_topic",
    "tone",
    "entities",
    "market_relevance",
    "policy_direction",
    "classification_reason",
    "classification_status",
    "classification_fallback",
    "classification_text_truncated",
]
MAX_CLASSIFICATION_TEXT_CHARS = 5000
DEFAULT_CLASSIFICATION_BATCH_SIZE = 10


class FatalClassificationError(RuntimeError):
    """Unrecoverable LLM configuration or account error."""


TOPIC_LABELS = [
    "tariff_trade",
    "oil_energy",
    "war_defense",
    "healthcare_pharma",
    "rates_usd",
    "crypto",
    "broad_market",
    "immigration_border",
    "legal_court",
    "election_politics",
    "self_promotion",
    "other",
]

TONE_LABELS = [
    "aggressive",
    "deescalating",
    "promotional",
    "neutral",
    "defensive",
    "threatening",
    "praising",
]

MARKET_RELEVANCE_LABELS = ["high", "medium", "low"]
POLICY_DIRECTION_LABELS = ["escalation", "deescalation", "easing", "uncertainty", "neutral"]


CLASSIFICATION_GUIDE = """
Decision style:
- Choose a specific topic when the post clearly supports one; use other for vague, ceremonial, or personal posts.
- Use neutral tone when the post is informational or too short to infer praise, attack, defense, threat, or promotion.
- Use low market_relevance for personal, ceremonial, sports/fitness, memorial, or generic political posts with no plausible market channel.
- Use neutral policy_direction for ceremonial actions, transparency/file releases, fitness/awareness programs, praise, condolences, or generic campaign posts.

Topic hints:
- tariff_trade=tariffs/trade deals/imports/exports; oil_energy=oil/gas/drilling/OPEC/energy prices.
- war_defense=war/military/NATO/Iran/Russia/Ukraine/Israel; healthcare_pharma=healthcare/drug prices/FDA/pharma.
- rates_usd=Fed/rates/inflation/dollar/Treasury/deficits; crypto=Bitcoin/crypto/digital assets.
- broad_market=stocks/economy/jobs/GDP/recession/business sentiment; immigration_border=border/deportation/asylum.
- legal_court=lawsuits/indictments/trials/judges/Supreme Court/DOJ/FBI; election_politics=campaigns/polls/voting/endorsements.
- self_promotion=rallies/media appearances/fundraising/personal brand when no stronger issue topic fits.

Market relevance guide:
- high: direct macro/market/sector signal, policy threat/action, geopolitical shock, major legal/election risk, or named market-sensitive entities.
- medium: indirect but plausible market channel through policy, politics, regulation, trade, legal risk, or sentiment.
- low: personal, ceremonial, or purely social content with no plausible market channel.

Policy direction guide:
- escalation: threats, new restrictions, tariffs, sanctions, military/legal pressure, crackdowns, or confrontation.
- deescalation: peace, deals, compromise, removed threats, reduced conflict.
- easing: tax cuts, deregulation, lower rates, subsidies, looser policy, pro-business relief.
- uncertainty: unclear, mixed, conditional, speculative, or contradictory policy direction.
- neutral: no policy/conflict direction.

Entities should include named countries, institutions, companies, people, sectors, commodities, or assets explicitly mentioned.
""".strip()


def build_classification_messages(cleaned_text: str, strict: bool = False) -> list[dict[str, str]]:
    strict_instruction = (
        "Your previous response was invalid. Return only valid JSON with exactly the required keys. "
        "Do not include markdown, commentary, code, SQL, or extra fields."
        if strict
        else "Return only valid JSON. Do not include markdown or commentary."
    )

    system_prompt = (
        "You classify Truth Social posts into a small market narrative schema. "
        "Use only the allowed labels. Do not choose tickers, write SQL, or analyze market prices. "
        "Prefer specific labels when supported, but use neutral, low, or other for vague or ceremonial posts."
    )
    user_prompt = f"""
Classify this post.

Allowed primary_topic labels: {", ".join(TOPIC_LABELS)}
Allowed tone labels: {", ".join(TONE_LABELS)}
Allowed market_relevance labels: {", ".join(MARKET_RELEVANCE_LABELS)}
Allowed policy_direction labels: {", ".join(POLICY_DIRECTION_LABELS)}

Classification guide:
{CLASSIFICATION_GUIDE}

Output schema:
{{
  "primary_topic": "one allowed primary_topic",
  "tone": "one allowed tone",
  "entities": ["short entity names"],
  "market_relevance": "one allowed market_relevance",
  "policy_direction": "one allowed policy_direction",
  "reason": "one short sentence"
}}

{strict_instruction}

Post:
{cleaned_text[:MAX_CLASSIFICATION_TEXT_CHARS]}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_batch_classification_messages(
    posts: list[tuple[str, str]],
    strict: bool = False,
) -> list[dict[str, str]]:
    strict_instruction = (
        "Your previous response was invalid. Return only valid JSON with exactly one classification "
        "for every input item_id. Do not include markdown, commentary, code, SQL, or extra fields."
        if strict
        else "Return only valid JSON. Do not include markdown or commentary."
    )

    system_prompt = (
        "You classify Truth Social posts into a small market narrative schema. "
        "Use only the allowed labels. Do not choose tickers, write SQL, or analyze market prices. "
        "Classify each post independently. Prefer specific labels when supported, but use neutral, "
        "low, or other for vague or ceremonial posts."
    )
    post_blocks = "\n\n".join(
        f"item_id: {item_id}\ntext: {text[:MAX_CLASSIFICATION_TEXT_CHARS]}"
        for item_id, text in posts
    )
    user_prompt = f"""
Classify each post below.

Allowed primary_topic labels: {", ".join(TOPIC_LABELS)}
Allowed tone labels: {", ".join(TONE_LABELS)}
Allowed market_relevance labels: {", ".join(MARKET_RELEVANCE_LABELS)}
Allowed policy_direction labels: {", ".join(POLICY_DIRECTION_LABELS)}

Classification guide:
{CLASSIFICATION_GUIDE}

Output schema:
{{
  "classifications": [
    {{
      "item_id": "same item_id from input",
      "primary_topic": "one allowed primary_topic",
      "tone": "one allowed tone",
      "entities": ["short entity names"],
      "market_relevance": "one allowed market_relevance",
      "policy_direction": "one allowed policy_direction",
      "reason": "one short sentence"
    }}
  ]
}}

Rules:
- Return exactly one classification for every input item_id.
- Do not combine posts.
- Do not invent item_id values.

{strict_instruction}

Posts:
{post_blocks}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced_match = JSON_BLOCK_PATTERN.search(text)
    if fenced_match:
        text = fenced_match.group(1).strip()
    else:
        text = extract_first_balanced_json_object(text)
    return json.loads(text)


def extract_first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text


def validate_classification(raw_output: str) -> NarrativeClassification:
    return NarrativeClassification.model_validate(extract_json_object(raw_output))


def validate_batch_classification(
    raw_output: str,
    expected_item_ids: set[str],
) -> dict[str, NarrativeClassification]:
    payload = extract_json_object(raw_output)
    raw_items = payload.get("classifications")
    if not isinstance(raw_items, list):
        raise ValueError("Batch classification response must include a classifications list.")

    classifications: dict[str, NarrativeClassification] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("Each batch classification must be a JSON object.")
        item_id = str(raw_item.get("item_id", "")).strip()
        if item_id not in expected_item_ids:
            raise ValueError(f"Unexpected batch classification item_id: {item_id}")
        if item_id in classifications:
            raise ValueError(f"Duplicate batch classification item_id: {item_id}")

        classification_payload = {key: value for key, value in raw_item.items() if key != "item_id"}
        classifications[item_id] = NarrativeClassification.model_validate(classification_payload)

    if set(classifications) != expected_item_ids:
        missing_ids = ", ".join(sorted(expected_item_ids.difference(classifications)))
        raise ValueError(f"Batch classification response is missing item_id values: {missing_ids}")
    return classifications


def classification_to_columns(classification: NarrativeClassification) -> dict[str, Any]:
    return {
        "primary_topic": classification.primary_topic.value,
        "tone": classification.tone.value,
        "entities": ",".join(classification.entities),
        "market_relevance": classification.market_relevance.value,
        "policy_direction": classification.policy_direction.value,
        "classification_reason": classification.reason,
    }


def classification_result_row(
    classification: NarrativeClassification,
    status: str,
    text_truncated: bool,
) -> dict[str, Any]:
    return {
        **classification_to_columns(classification),
        "classification_status": status,
        "classification_fallback": status != "ok",
        "classification_text_truncated": text_truncated,
    }


def classify_text(
    cleaned_text: str,
    llm: LLMCallable | None,
) -> tuple[NarrativeClassification, str]:
    if not cleaned_text.strip():
        return fallback_classification(), "empty_text"
    if llm is None:
        return fallback_classification(), "no_llm"

    for strict in (False, True):
        messages = build_classification_messages(cleaned_text, strict=strict)
        try:
            return validate_classification(llm(messages)), "ok"
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            continue
        except FatalClassificationError:
            raise
        except Exception:
            return fallback_classification(), "llm_error"

    return fallback_classification(), "invalid_output"


def classify_text_batch(
    texts: list[str],
    llm: LLMCallable | None,
) -> list[tuple[NarrativeClassification, str]]:
    results: list[tuple[NarrativeClassification, str] | None] = [None] * len(texts)
    non_empty_posts: list[tuple[str, str]] = []

    for position, cleaned_text in enumerate(texts):
        if not cleaned_text.strip():
            results[position] = (fallback_classification(), "empty_text")
        elif llm is None:
            results[position] = (fallback_classification(), "no_llm")
        else:
            non_empty_posts.append((str(position), cleaned_text))

    if non_empty_posts:
        expected_item_ids = {item_id for item_id, _ in non_empty_posts}
        for strict in (False, True):
            messages = build_batch_classification_messages(non_empty_posts, strict=strict)
            try:
                classifications = validate_batch_classification(llm(messages), expected_item_ids)
                for item_id, classification in classifications.items():
                    results[int(item_id)] = (classification, "ok")
                break
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
                if strict:
                    for item_id, _ in non_empty_posts:
                        results[int(item_id)] = (fallback_classification(), "invalid_output")
                continue
            except FatalClassificationError:
                raise
            except Exception:
                for item_id, _ in non_empty_posts:
                    results[int(item_id)] = (fallback_classification(), "llm_error")
                break

    return [result if result is not None else (fallback_classification(), "invalid_output") for result in results]


def batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def iter_with_progress(values: pd.Series, show_progress: bool) -> Any:
    if not show_progress:
        return values
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, desc="classify", total=len(values))


def validate_no_existing_classification_columns(df: pd.DataFrame) -> None:
    overlapping = [column for column in CLASSIFICATION_COLUMNS if column in df.columns]
    if overlapping:
        overlap_text = ", ".join(overlapping)
        raise ValueError(f"Input already contains classification columns: {overlap_text}")


def build_classification_frame(
    source_df: pd.DataFrame,
    llm: LLMCallable | None,
    limit: int | None,
    show_progress: bool,
    batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
) -> pd.DataFrame:
    rows_to_classify = source_df.head(limit) if limit is not None else source_df
    output_rows: list[dict[str, Any]] = []
    cleaned_text_values = rows_to_classify["cleaned_text"].fillna("").astype(str)

    text_batches = batched(cleaned_text_values.tolist(), batch_size)
    for text_batch in iter_with_progress(text_batches, show_progress=show_progress):
        batch_results = classify_text_batch(text_batch, llm)
        for cleaned_text, (classification, status) in zip(text_batch, batch_results):
            output_rows.append(
                classification_result_row(
                    classification=classification,
                    status=status,
                    text_truncated=len(cleaned_text) > MAX_CLASSIFICATION_TEXT_CHARS,
                )
            )

    classified = pd.DataFrame(output_rows, index=rows_to_classify.index)

    if limit is not None and limit < len(source_df):
        fallback = classification_result_row(
            fallback_classification(),
            status="not_processed",
            text_truncated=False,
        )
        remaining_index = source_df.index.difference(classified.index)
        remaining = pd.DataFrame([fallback] * len(remaining_index), index=remaining_index)
        classified = pd.concat([classified, remaining]).sort_index()

    classified["classification_fallback"] = classified["classification_fallback"].astype(bool)
    classified["classification_text_truncated"] = classified["classification_text_truncated"].astype(bool)
    return classified


def classify_dataframe(
    df: pd.DataFrame,
    llm: LLMCallable | None,
    limit: int | None = None,
    show_progress: bool = False,
    batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
) -> pd.DataFrame:
    if "cleaned_text" not in df.columns:
        raise ValueError("Input data must include a cleaned_text column.")
    validate_no_existing_classification_columns(df)

    classified = build_classification_frame(
        df,
        llm=llm,
        limit=limit,
        show_progress=show_progress,
        batch_size=batch_size,
    )
    return pd.concat([df.reset_index(drop=True), classified.reset_index(drop=True)], axis=1)


def merge_existing_classifications(df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns or "post_id" not in existing_df.columns:
        raise ValueError("Resume requires post_id in both input and existing output.")

    missing_columns = [
        column for column in CLASSIFICATION_COLUMNS if column not in existing_df.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Existing output is missing classification columns: {missing_text}")

    existing = existing_df[["post_id", *CLASSIFICATION_COLUMNS]].copy()
    existing["post_id"] = existing["post_id"].astype(str)
    existing = existing.drop_duplicates("post_id", keep="last").set_index("post_id")

    result = df.copy()
    result["post_id"] = result["post_id"].astype(str)
    for column in CLASSIFICATION_COLUMNS:
        result[column] = result["post_id"].map(existing[column])
    return result


def unclassified_mask(df: pd.DataFrame) -> pd.Series:
    return df["primary_topic"].isna()


def write_checkpoint(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, output_path)


def classify_dataframe_incremental(
    df: pd.DataFrame,
    llm: LLMCallable | None,
    output_path: Path,
    limit: int | None = None,
    checkpoint_every: int = 1000,
    resume: bool = False,
    show_progress: bool = True,
    batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
) -> pd.DataFrame:
    if "cleaned_text" not in df.columns:
        raise ValueError("Input data must include a cleaned_text column.")

    source = df.copy()
    if resume and output_path.exists():
        source = merge_existing_classifications(source, pd.read_parquet(output_path))
    else:
        validate_no_existing_classification_columns(source)
        for column in CLASSIFICATION_COLUMNS:
            source[column] = pd.NA

    candidate_index = source.index[unclassified_mask(source)]
    if limit is not None:
        candidate_index = candidate_index[:limit]

    index_batches = batched(list(candidate_index), batch_size)
    processed_since_checkpoint = 0

    for index_batch in iter_with_progress(index_batches, show_progress=show_progress):
        text_batch = source.loc[index_batch, "cleaned_text"].fillna("").astype(str).tolist()
        batch_results = classify_text_batch(text_batch, llm)

        for index, cleaned_text, (classification, status) in zip(
            index_batch,
            text_batch,
            batch_results,
        ):
            row = classification_result_row(
                classification=classification,
                status=status,
                text_truncated=len(cleaned_text) > MAX_CLASSIFICATION_TEXT_CHARS,
            )
            for column, value in row.items():
                source.at[index, column] = value

        processed_since_checkpoint += len(index_batch)
        if checkpoint_every > 0 and processed_since_checkpoint >= checkpoint_every:
            write_checkpoint(source, output_path)
            processed_since_checkpoint = 0

    remaining = unclassified_mask(source)
    if remaining.any():
        fallback = classification_result_row(
            fallback_classification(),
            status="not_processed",
            text_truncated=False,
        )
        for column, value in fallback.items():
            source.loc[remaining, column] = value

    source["classification_fallback"] = source["classification_fallback"].astype(bool)
    source["classification_text_truncated"] = source["classification_text_truncated"].astype(bool)
    write_checkpoint(source, output_path)
    return source


def create_openai_llm(model: str = DEFAULT_CLASSIFICATION_MODEL) -> LLMCallable | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    effective_model = os.getenv("OPENAI_CLASSIFICATION_MODEL", model)

    try:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            BadRequestError,
            OpenAI,
            RateLimitError,
        )
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    retry_exceptions = (RateLimitError, APITimeoutError, APIConnectionError, APIError)

    def describe_openai_error(error: Exception) -> str:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            error_body = body.get("error", body)
            message = error_body.get("message")
            code = error_body.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
        return str(error)

    def is_insufficient_quota_error(error: Exception) -> bool:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            error_body = body.get("error", body)
            code = str(error_body.get("code", "")).lower()
            message = str(error_body.get("message", "")).lower()
            return code == "insufficient_quota" or "insufficient quota" in message
        return "insufficient_quota" in str(error).lower()

    def call_openai(messages: list[dict[str, str]]) -> str:
        for attempt in range(4):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": effective_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "timeout": 30,
                }
                if supports_zero_temperature(effective_model):
                    request_kwargs["temperature"] = 0
                response = client.chat.completions.create(
                    **request_kwargs,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI returned an empty classification response.")
                return content
            except BadRequestError as error:
                detail = describe_openai_error(error)
                raise FatalClassificationError(
                    f"OpenAI classification request is invalid: {detail}"
                ) from error
            except retry_exceptions as error:
                if is_insufficient_quota_error(error):
                    detail = describe_openai_error(error)
                    raise FatalClassificationError(
                        f"OpenAI classification cannot continue because quota is unavailable: {detail}"
                    ) from error
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("OpenAI classification failed after retries.")

    return call_openai


def supports_zero_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


def openai_llm_unavailable_reason() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set."
    if importlib.util.find_spec("openai") is None:
        return "the openai package is not installed in this Python environment."
    return "the OpenAI client could not be initialized."


def run_classification(
    input_path: Path = CLEANED_EVENTS_PATH,
    output_path: Path = CLASSIFIED_EVENTS_PATH,
    model: str = DEFAULT_CLASSIFICATION_MODEL,
    limit: int | None = None,
    fallback_only: bool = False,
    checkpoint_every: int = 1000,
    resume: bool = False,
    progress: bool = True,
    batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    llm = None if fallback_only else create_openai_llm(model)
    if llm is None and not fallback_only:
        reason = openai_llm_unavailable_reason()
        raise RuntimeError(
            f"OpenAI live classification is unavailable: {reason} "
            "Install dependencies and set OPENAI_API_KEY, or pass --fallback-only for an intentional fallback run."
        )
    classified_df = classify_dataframe_incremental(
        df,
        llm=llm,
        output_path=output_path,
        limit=limit,
        checkpoint_every=checkpoint_every,
        resume=resume,
        show_progress=progress,
        batch_size=batch_size,
    )
    return classified_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify cleaned market narrative events.")
    parser.add_argument("--input-path", type=Path, default=CLEANED_EVENTS_PATH)
    parser.add_argument("--output-path", type=Path, default=CLASSIFIED_EVENTS_PATH)
    parser.add_argument("--model", default=DEFAULT_CLASSIFICATION_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CLASSIFICATION_BATCH_SIZE,
        help="Number of posts to classify per LLM request. Use 1 for single-post requests.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip LLM calls and write validated fallback classifications.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        classified_df = run_classification(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            limit=args.limit,
            fallback_only=args.fallback_only,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            progress=not args.no_progress,
            batch_size=args.batch_size,
        )
    except FatalClassificationError as error:
        raise SystemExit(f"Classification stopped: {error}") from error
    fallback_count = int(classified_df["classification_fallback"].sum())
    print(
        f"Wrote {len(classified_df):,} rows to {args.output_path} "
        f"({fallback_count:,} fallback classifications)"
    )


if __name__ == "__main__":
    main()
