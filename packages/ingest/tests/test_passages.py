"""Reading an original: its text, its passages and its date."""

from __future__ import annotations

import pytest

from prose import pdf
from spatial_ingest.rules.passages import (
    UnreadableSource,
    chunk,
    last_change,
    text_of,
)

HTML = "text/html; charset=utf-8"
PAGE = """<!doctype html>
<html lang="de"><head><title>KIT</title><style>p { color: red }</style></head>
<body>
<nav><a href="/">Startseite</a></nav>
<!-- CONTENT START -->
<h3>30.22 Gaede Hs</h3>
<ul>
  <li><span class="title">Stockwerk:</span><span>2. OG</span></li>
  <li><span class="title">Zugang zum HS:</span><span>&uuml;ber den Aufzug</span></li>
</ul>
<script>window.location.href = "/next"</script>
<p>Der H&ouml;rsaal hat 2 Stufen.</p>
<!-- CONTENT END -->
<footer><span class="last_change">letzte Änderung:
    31.05.2012</span></footer>
</body></html>
""".encode()


def test_a_page_reads_as_what_a_reader_sees() -> None:
    text = text_of(PAGE, HTML)

    assert "Startseite" in text
    assert "30.22 Gaede Hs" in text
    assert "Stockwerk: 2. OG" in text
    assert "Zugang zum HS: über den Aufzug" in text
    assert "Der Hörsaal hat 2 Stufen." in text
    assert "window.location" not in text
    assert "color: red" not in text


def test_markers_keep_the_navigation_and_footer_out() -> None:
    text = text_of(PAGE, HTML, between=("<!-- CONTENT START -->", "<!-- CONTENT END -->"))

    assert text == (
        "30.22 Gaede Hs\n\n"
        "Stockwerk: 2. OG\nZugang zum HS: über den Aufzug\n\n"
        "Der Hörsaal hat 2 Stufen."
    )


def test_a_page_without_the_markers_is_read_whole() -> None:
    assert "Startseite" in text_of(PAGE, HTML, between=("<!-- NOPE -->", "<!-- NEVER -->"))


def test_a_pdf_is_read_whatever_the_server_calls_it() -> None:
    manual = pdf("Replace the lamp every 12 months.", "Switch off the breaker first.")

    for content_type in ("application/pdf", "application/octet-stream", ""):
        text = text_of(manual, content_type)
        assert "Replace the lamp every 12 months." in text
        assert "Switch off the breaker first." in text


def test_plain_text_is_tidied_and_keeps_its_paragraphs() -> None:
    text = text_of(b"One  line\r\nand another\r\n\r\n\r\nNext   paragraph\n", "text/plain")

    assert text == "One line\nand another\n\nNext paragraph"


def test_a_source_that_is_neither_is_unreadable() -> None:
    with pytest.raises(UnreadableSource):
        text_of(b"\x89PNG\r\n", "image/png")
    with pytest.raises(UnreadableSource):
        text_of(b"%PDF-1.4 and then nothing a reader could use", "application/pdf")


def test_short_paragraphs_share_a_passage() -> None:
    assert chunk("A heading\n\nOne line.\n\nAnother.", limit=40) == [
        "A heading\n\nOne line.\n\nAnother."
    ]


def test_a_passage_never_splits_a_paragraph_that_fits() -> None:
    first, second, third = "a" * 30, "b" * 30, "c" * 30

    assert chunk(f"{first}\n\n{second}\n\n{third}", limit=70) == [f"{first}\n\n{second}", third]


def test_a_paragraph_over_the_limit_is_cut_at_sentence_ends() -> None:
    sentences = ["This is sentence number one.", "Here is the second one.", "And a third."]

    passages = chunk(" ".join(sentences), limit=55)

    assert passages == ["This is sentence number one. Here is the second one.", "And a third."]
    assert all(len(passage) <= 55 for passage in passages)


def test_text_with_no_break_at_all_is_still_cut_to_the_limit() -> None:
    assert chunk("x" * 25, limit=10) == ["x" * 10, "x" * 10, "x" * 5]


def test_empty_text_has_no_passages() -> None:
    assert chunk("") == []


def test_the_date_is_read_from_the_last_change_line() -> None:
    assert last_change(PAGE, HTML) == "2012-05-31"
    assert last_change(b"<html><p>Last change: 3.2.2024</p></html>", HTML) == "2024-02-03"


@pytest.mark.parametrize(
    "page",
    [
        b"<html><p>Stand 2012, zuletzt gesehen 31.05.2012</p></html>",
        b"<html><p>letzte \xc3\x84nderung: 31.13.2012</p></html>",
        b"<html><p>letzte \xc3\x84nderung: neulich</p></html>",
    ],
)
def test_a_page_that_does_not_say_is_undated(page: bytes) -> None:
    assert last_change(page, HTML) is None


def test_only_a_page_is_dated() -> None:
    assert last_change(pdf("letzte Aenderung: 31.05.2012"), "application/pdf") is None
    assert last_change("letzte Änderung: 31.05.2012".encode(), "text/plain") is None
