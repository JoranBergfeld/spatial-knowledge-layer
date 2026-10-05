"""strip-person-names: takes the names of people working in a room out of its text.

The rule, as decided on the map (personal-website#50, extended to descriptions in #48):

1. An academic title (Prof., Dr., Dipl., PD.), any further title abbreviations
   hanging off it (Dr.-Ing., habil., rer. nat.), and the capitalised words after it.
2. Büro, Sekretariat, Labor or AG followed by a capitalised word. The keyword
   stays, the words go.

After a title every capitalised word goes, joined by "und" or "&" where there
are two people. After a keyword "capitalised" means mixed case, so "Labor II"
and "Sekretariat ITM" survive.
German capitalises every noun, so part 2 will also eat things like
"Sekretariat Geschäftsführung". That errs on the side of printing less, which is
the side we want. A surname that slips through is fixed by widening the rule,
never by editing one room.

Rooms named after someone (Gaede-Hörsaal) carry no title and no keyword, so they
pass through untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RULE = "strip-person-names@1"

_LETTERS = r"[^\W\d_]+"
_TITLE = re.compile(rf"\b(?:Prof|Dr|Dipl|PD)\.(?:[-\s]*{_LETTERS}\.)*")
_KEYWORD = re.compile(r"\b(?:Büro|Sekretariat|Labor|AG)\b:?")
# A word of a name, with the particles and joiners that may come before it.
_NAME_WORD = re.compile(
    rf"\s*(?:(?:und|&)\s+)?(?:(?:von|van|de|der|zu)\s+)*({_LETTERS}(?:[-'’]{_LETTERS})*)"
)

_EDGE_JUNK = " \t,;:/-&"


@dataclass(frozen=True)
class Filtered:
    # None when nothing is left to print.
    text: str | None
    changed: bool


def _end_of_name(text: str, start: int, *, mixed_case_only: bool) -> int:
    """Where the run of name words beginning at `start` stops."""
    end = start
    while word := _NAME_WORD.match(text, end):
        first = word.group(1)
        if not first[0].isupper() or (mixed_case_only and first.isupper()):
            break
        if text.startswith(".", word.end()):
            # An abbreviation, most likely the next person's title. Not a name.
            break
        end = word.end()
    return end


def _cut(text: str, opener: re.Pattern[str], *, keep_opener: bool) -> str:
    kept: list[str] = []
    position = 0
    while match := opener.search(text, position):
        end = _end_of_name(text, match.end(), mixed_case_only=keep_opener)
        kept.append(text[position : match.end() if keep_opener else match.start()])
        position = max(end, match.end())
    kept.append(text[position:])
    return "".join(kept)


def strip_person_names(text: str) -> Filtered:
    stripped = _cut(text, _TITLE, keep_opener=False)
    stripped = _cut(stripped, _KEYWORD, keep_opener=True)
    # "Büro (Prof. X)" leaves empty brackets behind.
    stripped = re.sub(r"\(\s*\)", "", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(_EDGE_JUNK)
    if stripped == text.strip():
        return Filtered(text=text, changed=False)
    return Filtered(text=stripped or None, changed=True)
