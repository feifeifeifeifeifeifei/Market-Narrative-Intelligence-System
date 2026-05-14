from src.clean import build_cleaned_text


def test_build_cleaned_text_prefers_plain_text_and_strips_urls() -> None:
    assert build_cleaned_text("Visit https://example.com now", "<p>Fallback</p>") == "Visit now"


def test_build_cleaned_text_falls_back_to_html() -> None:
    assert build_cleaned_text("", "<p>Hello&nbsp;<strong>world</strong></p>") == "Hello world"
