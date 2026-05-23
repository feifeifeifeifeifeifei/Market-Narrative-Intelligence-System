from src.clean import build_cleaned_text, build_retrieval_text


def test_build_cleaned_text_prefers_plain_text_and_strips_urls() -> None:
    assert build_cleaned_text("Visit https://example.com now", "<p>Fallback</p>") == "Visit now"


def test_build_cleaned_text_falls_back_to_html() -> None:
    assert build_cleaned_text("", "<p>Hello&nbsp;<strong>world</strong></p>") == "Hello world"


def test_build_retrieval_text_removes_retweet_shell_and_keeps_content() -> None:
    assert (
        build_retrieval_text("RT @realDonaldTrumpThe Iranian leadership forced Ships toward Texas oil")
        == "The Iranian leadership forced Ships toward Texas oil"
    )
    assert build_retrieval_text("RT @realDonaldTrump: Drill baby drill") == "Drill baby drill"


def test_build_retrieval_text_drops_pure_retweet_shell() -> None:
    assert build_retrieval_text("RT @realDonaldTrump") == ""


def test_build_retrieval_text_does_not_strip_mid_sentence_mentions() -> None:
    assert (
        build_retrieval_text("Energy policy from @realDonaldTrump moved markets")
        == "Energy policy from @realDonaldTrump moved markets"
    )
