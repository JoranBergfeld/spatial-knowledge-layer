"""Reading a document's original: its text, its passages and its date.

    text-from-source@1      the text of an HTML page, a PDF or a plain text file
    chunk-by-paragraph@1    whole paragraphs packed into passages of at most
                            1200 characters
    date-from-last-change@1 the day a page's own footer says it last changed

HTML keeps what a reader sees: no scripts, no styles, one line per block
element. A corpus can name two markers, and then only what lies between them is
read, which is how a page's navigation and footer stay out of the text.

The date rule reads one thing, the "letzte Änderung: 31.05.2012" (or "last
change:") line KIT's pages carry. It looks at the whole page, footer included.
No other rule dates anything, so a manual is undated. The day a document was
fetched is never its date.
"""

from __future__ import annotations

import datetime
import io
import re
from html.parser import HTMLParser

from pypdf import PdfReader
from pypdf.errors import PyPdfError

TEXT_RULE = "text-from-source@1"
CHUNK_RULE = "chunk-by-paragraph@1"
DATE_RULE = "date-from-last-change@1"

# A passage's longest length in characters. A paragraph longer than this is cut
# at line ends, then at sentence ends.
CHUNK_LIMIT = 1200

_SKIPPED = {"script", "style", "noscript", "head", "template", "svg"}
# Elements that start a new paragraph, and elements that only start a new line.
_PARAGRAPHS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}
_PARAGRAPHS |= {"ul", "ol", "dl", "blockquote", "pre", "hr"}
_LINES = {"br", "li", "div", "tr", "dt", "dd"}
_LAST_CHANGE = re.compile(
    r"(?:letzte\s+Änderung|last\s+change)\s*:?\s*(?:<[^>]*>\s*)*(\d{1,2})\.(\d{1,2})\.(\d{4})",
    re.IGNORECASE,
)


# Stand-ins for a paragraph break and a line break while a page is being read.
_BREAK, _LINE = "\x1e", "\x1f"


def _layout(run: re.Match[str]) -> str:
    return "\n\n" if _BREAK in run.group() else "\n" if _LINE in run.group() else " "


class UnreadableSource(Exception):
    pass


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED:
            self._skipping += 1
        self._boundary(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED and self._skipping:
            self._skipping -= 1
        self._boundary(tag)

    def _boundary(self, tag: str) -> None:
        # A space at every tag, so "<span>Floor:</span><span>2</span>" reads "Floor: 2".
        self.parts.append(_BREAK if tag in _PARAGRAPHS else _LINE if tag in _LINES else " ")

    def handle_data(self, data: str) -> None:
        # A line end in the markup means nothing. Only the elements break lines.
        if not self._skipping:
            self.parts.append(" ".join(data.split()) if data.strip() else " ")


def _tidy(text: str) -> str:
    """One space between words, no empty lines inside a paragraph, one between paragraphs."""
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [" ".join(line.split()) for line in paragraph.split("\n")]
        kept = [line for line in lines if line]
        if kept:
            paragraphs.append("\n".join(kept))
    return "\n\n".join(paragraphs)


def html_text(page: str, between: tuple[str, str] | None = None) -> str:
    if between is not None:
        start, end = between
        if start in page and end in page.split(start, 1)[1]:
            page = page.split(start, 1)[1].split(end, 1)[0]
    parser = _Text()
    parser.feed(page)
    parser.close()
    return _tidy(re.sub(r"[\s\x1e\x1f]+", _layout, "".join(parser.parts)))


def pdf_text(content: bytes) -> str:
    try:
        pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages]
    except (PyPdfError, ValueError, KeyError, OSError) as error:
        raise UnreadableSource(f"not a readable PDF: {error}") from error
    return _tidy("\n\n".join(pages))


def kind_of(content: bytes, content_type: str) -> str:
    """html, pdf or text, by what the bytes are and then by what the server said."""
    declared = content_type.split(";")[0].strip().lower()
    if content.lstrip()[:5] == b"%PDF-" or declared == "application/pdf":
        return "pdf"
    if "html" in declared or content.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        return "html"
    if declared.startswith("text/") or not declared:
        return "text"
    raise UnreadableSource(f"cannot read a {declared} source")


def _decoded(content: bytes, content_type: str) -> str:
    charset = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    for encoding in ([charset.group(1)] if charset else []) + ["utf-8"]:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("latin-1")


def text_of(content: bytes, content_type: str, between: tuple[str, str] | None = None) -> str:
    """The readable text of a fetched original."""
    kind = kind_of(content, content_type)
    if kind == "pdf":
        return pdf_text(content)
    decoded = _decoded(content, content_type)
    return html_text(decoded, between) if kind == "html" else _tidy(decoded)


def _pieces(paragraph: str, limit: int) -> list[str]:
    """A paragraph over the limit, cut at line ends, then sentence ends, then anywhere."""
    if len(paragraph) <= limit:
        return [paragraph]
    for separator, pattern in (("\n", r"\n"), (" ", r"(?<=[.!?])\s+")):
        units = re.split(pattern, paragraph)
        if len(units) > 1:
            packed: list[str] = []
            for unit in units:
                if packed and len(packed[-1]) + len(separator) + len(unit) <= limit:
                    packed[-1] += separator + unit
                else:
                    packed.append(unit)
            return [piece for unit in packed for piece in _pieces(unit, limit)]
    return [paragraph[start : start + limit] for start in range(0, len(paragraph), limit)]


def chunk(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """The text as passages: whole paragraphs, as many as fit under the limit."""
    passages: list[str] = []
    for paragraph in text.split("\n\n"):
        for piece in _pieces(paragraph.strip(), limit):
            if not piece:
                continue
            if passages and len(passages[-1]) + 2 + len(piece) <= limit:
                passages[-1] += "\n\n" + piece
            else:
                passages.append(piece)
    return passages


def last_change(content: bytes, content_type: str) -> str | None:
    """The ISO date a page says it last changed, or None when it does not say."""
    if kind_of(content, content_type) != "html":
        return None
    found = _LAST_CHANGE.search(_decoded(content, content_type))
    if not found:
        return None
    day, month, year = (int(part) for part in found.groups())
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None
