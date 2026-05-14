import html
import re
from typing import Any

from bs4 import BeautifulSoup


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\ufeff]")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if value != value:
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def html_to_text(value: Any) -> str:
    if is_blank(value):
        return ""
    soup = BeautifulSoup(str(value), "html.parser")
    return soup.get_text(" ", strip=True)


def normalize_post_text(value: Any) -> str:
    if is_blank(value):
        return ""
    cleaned = html.unescape(str(value))
    if "<" in cleaned and ">" in cleaned:
        cleaned = BeautifulSoup(cleaned, "html.parser").get_text(" ", strip=True)
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = cleaned.replace("�", "")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def build_cleaned_text(text: Any, content_html: Any) -> str:
    source = text if not is_blank(text) else html_to_text(content_html)
    return normalize_post_text(source)
