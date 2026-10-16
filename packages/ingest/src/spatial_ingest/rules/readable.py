"""readable-passage: a passage that is mostly OCR junk never becomes a chunk.

The test bench's O&M documents are mostly scans. Tesseract reads prose fine, and
reads drawings, wiring diagrams and upside-down pages as junk like this:

    YN HWP—two | 5-2 Ome J A ra C) ) ET-2/ . 80% | BOILER HWS _ Iw 100¢ B-1 ...
    Ayenuuy / Ajjenuury-ILWwasg VS / Apayend re) IUBYDSP\\ 48]Se)\\ W ...

Retrieval returns that, the composer cites it and the NLI check drops it (#31).
So every passage is read token by token, a token being what lies between spaces,
and a passage where too many tokens are bad is dropped before it is translated,
embedded or indexed.

A token is fine when, with the punctuation around it taken off, it is

- a word: letters only, joined by a hyphen, an apostrophe or a slash at most,
  in one case pattern (lower, UPPER or Capitalised). "a", "A" and "I" are the
  only one-letter words. A word of four letters or more in lower or Capitalised
  case with a letter pair English hardly has ("Ayenuuy" has "uu") is bad, in an
  English passage only
- a number, with a unit or a fraction stuck to it if it has one: "6,051mm",
  "3/8", "1-15/16", "4½", "80%"
- a code: letters and digits together, joined by hyphens, dots or slashes:
  "B-1", "P-9050M", "026-32401-297"

Anything else is bad: stray symbols ("|", "_", "~-~~~I"), single letters other
than the three words, mixed-case letters ("weYD", "SuNnssadd"), letters with
symbols inside ("C)", "100¢"). Punctuation on its own does not count either way.

A passage with at least BAD_SHARE of its tokens bad is dropped. Tables of codes
and numbers pass: they are what a plumbing schedule is. A passage with fewer than
FEW tokens is never dropped, there is too little of it to judge.

The letter pairs are the ones fewer than 61 of the 235,886 words in the system
dictionary (/usr/share/dict/words, Webster's 2nd) have, minus "uh". They are a
list, so English that uses one a lot gets a passage dropped that should not be.
Fix that by taking the pair out.
"""

from __future__ import annotations

import re

RULE = "readable-passage@1"

# The share of bad tokens at which a passage is dropped.
BAD_SHARE = 0.25
# Fewer tokens than this and a passage is kept whatever they are.
FEW = 8

_RARE_PAIRS = frozenset(
    """
    bg bk bq bw bx bz cb cd cf cg cj cm cp cv cw cx cz dk dq dx dz fb fc fd fg fh fj fk fm fn
    fp fq fs fv fw fx fz gc gj gk gp gq gv gx gz hg hj hk hq hv hx hz iy jb jc jd jf jg jh jj
    jk jl jm jn jp jq jr js jt jv jw jx jy jz kc kd kg kj kq kv kx kz lj lq lx lz mc md mg mh
    mj mk mq mt mv mx mz pd pg pj pk pq pv px pz qa qb qc qd qe qf qg qh qi qj qk ql qm qn qo
    qp qq qr qs qt qv qw qx qy qz rx sj sx sz tj tk tq tv tx uj uq uu uw vb vc vd vf vg vh vj
    vk vl vm vn vp vq vr vs vt vv vw vx vz wc wg wj wp wq wv ww wx wz xb xd xf xg xj xk xl xm
    xn xq xr xs xv xw xx xz yj yk yq yv yy zb zc zd zf zg zh zj zk zm zn zp zq zr zs zt zv zw
    zx
    """.split()
)

# What is taken off either end of a token before it is judged.
_EDGES = re.compile(r"^[\"'“”‘’«»„‚(\[{<.,;:!?*•·…]+|[\"'“”‘’«»„‚)\]}>.,;:!?*•·…]+$")
# Tokens that are punctuation in running text: bullets and dashes. Any other
# token with no letter or digit in it ("|", "_", "~-~~~") is bad.
_PUNCTUATION = frozenset("•●■▪◆►○–—-&/+=#§")
_LETTERS = r"[^\W\d_]+"
# Inch and foot marks, and the marks a product name carries.
_MARKS = '″"”˝′®™'
_WORD = re.compile(rf"{_LETTERS}(?:[-'’/]{_LETTERS})*[{_MARKS}]?")
# Words the PDF's text layer glued together: "TemperatureRange", "TradeMaster".
_GLUED = re.compile(rf"[A-Z]?[a-z]{{2,}}(?:[A-Z][a-z]{{2,}})+[{_MARKS}]?")
# An acronym with a plural s: "GFCIs".
_PLURAL_ACRONYM = re.compile(r"[A-Z]{2,}s")
# Abbreviations with dots: "P.O.", "S.A.", "e.g.". The edges took the last dot off.
_DOTTED = re.compile(r"(?:[^\W\d_]\.){1,4}[^\W\d_]?")
_ADDRESS = re.compile(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_FRACTION = "½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
_NUMBER = re.compile(
    rf"[-+±~$€£]?(?:\d[\d.,:/⁄×x-]*[{_FRACTION}]?|[{_FRACTION}])"
    r"[%°″′\"”˝]?[^\W\d_]{0,4}[²³]?"
    r"|[°º][CFK]"
)
_CODE = re.compile(r"(?=.*\d)(?=.*[^\W\d_])[^\W_]+(?:[-./#][^\W_]+)*")
# One letter on its own is a word only as one of these. "x" is the one in "12 x 34".
_ONE_LETTER = frozenset({"a", "A", "x"})
# A word this short is fine in any case: units like "pH", "mV", "kW".
_SHORT = 3


def _one_case(word: str) -> bool:
    return all(
        part.islower() or part.isupper() or (part[0].isupper() and part[1:].islower())
        for part in re.findall(_LETTERS, word)
    )


def _rare_pair(word: str) -> bool:
    if len(word) < 4 or word.isupper():
        return False
    folded = word.lower()
    return any(folded[i : i + 2] in _RARE_PAIRS for i in range(len(folded) - 1))


def _bad(token: str, english: bool) -> bool | None:
    """Whether a token is bad. None for punctuation, which does not count."""
    bare = _EDGES.sub("", token)
    if not bare or bare in _PUNCTUATION:
        return None
    if _NUMBER.fullmatch(bare) or _ADDRESS.fullmatch(bare) or _DOTTED.fullmatch(bare):
        return False
    if _PLURAL_ACRONYM.fullmatch(bare):
        return False
    if _GLUED.fullmatch(bare):
        return english and _rare_pair(bare)
    if _WORD.fullmatch(bare):
        if len(bare) == 1:
            return bare not in _ONE_LETTER
        if len(bare.rstrip(_MARKS)) <= _SHORT:
            return False
        return not _one_case(bare) or (english and _rare_pair(bare))
    return not _CODE.fullmatch(bare)


def bad_share(text: str, english: bool = True) -> float:
    """The share of the passage's tokens that are bad, punctuation alone not counted."""
    judged = [bad for bad in (_bad(token, english) for token in text.split()) if bad is not None]
    return sum(judged) / len(judged) if judged else 0.0


def readable(text: str, english: bool = True) -> bool:
    """Whether a passage is kept."""
    if sum(_bad(token, english) is not None for token in text.split()) < FEW:
        return True
    return bad_share(text, english) < BAD_SHARE
