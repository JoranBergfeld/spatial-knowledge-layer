"""numbers-in-words: small counts, clock times and floors as words in ingested English text.

The entailment model that checks answers against passages misses on digits: the
two misses measured on translated KIT pages were a floor and a time
(personal-website#38). So build rewrites the kinds of number an answer is likely
to repeat, and leaves everything else as the source has it.

1. A clock time, hours and minutes on the 24 hour clock: "8:00" is "eight
   o'clock", "16:30" is "sixteen thirty", "18:05" is "eighteen oh five".
2. A floor ordinal: "2nd floor" is "second floor", "floor 2" is "floor two".
   Only next to floor, storey, story or level. "The 3rd edition" stays.
3. A small standalone count, 0 to 20: "2 steps" is "two steps".

Left as digits, always:

- a measured value with its unit: "90 cm", "37.1 MHz", "6 %", "5 minutes"
- anything over 20, and anything that is part of a longer token: "10.11",
  "608-42812", "31.05.2012", "1-3", "1/2", "B1"
- a number that names something: after a label word ("room 7", "channel 2",
  "page 3"), after an abbreviation ("Tel. 5", "Kaiserstr. 12") or after a street
  name ("Engesserstraße 5")
- a ratio that looks like a time, after gradient, slope, ratio, scale or incline

It only reads English. German ordinals ("2. OG") are the translation's job.
The label words and units are lists, so one that is missing gets a number
written out that should not be. Fix that by adding to the list.
"""

from __future__ import annotations

import re

RULE = "numbers-in-words@1"

# The largest count written out. Past this, a number is more figure than count.
SMALL = 20

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}
_ORDINALS = (
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
    "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
    "twentieth"
).split()

# Unit symbols match case as written. Unit words match in any case.
_UNIT_SYMBOLS = (
    "mm cm dm m km m² m2 m³ m3 g kg t ml l Hz kHz MHz GHz V kV mV A mA W kW VA kVA Wh kWh "
    "lx lm cd dB Pa kPa hPa bar N Nm rpm min h s ms K"
).split()
_UNIT_WORDS = (
    "millimetre millimeter centimetre centimeter metre meter kilometre kilometer inch inches "
    "foot feet gram kilogram tonne litre liter second minute hour day week month year "
    "percent degree volt amp ampere watt hertz decibel lux"
).split()
_UNIT = re.compile(
    r"\s?(?:[%°]|(?:"
    + "|".join(re.escape(symbol) for symbol in sorted(_UNIT_SYMBOLS, key=len, reverse=True))
    + r")(?![\w])|(?i:"
    + "|".join(_UNIT_WORDS)
    + r")(?i:s|es)?(?![\w]))"
)

_FLOOR = r"(?:floor|storey|story|level)"
# Words that make the number after them a name, not a count.
_LABELS = (
    "room hall building channel no nr number page figure fig table section chapter type model "
    "line route bus tram zone class version part item step steps gate exit entrance door lift "
    "elevator hs hörsaal raum gebäude kanal"
).split()
_STREET = re.compile(r"(?i)(?:str|straße|strasse|street|road|weg|platz|allee|ring|damm)$")
_RATIO_WORDS = {"gradient", "slope", "ratio", "scale", "incline"}

_TIME = re.compile(r"(?<![\w.,:/+-])([01]?\d|2[0-3]):([0-5]\d)(?![\w]|[.,:/-]\d)")
_ORDINAL = re.compile(
    rf"(?<![\w.,:/+-])(\d{{1,2}})(?:st|nd|rd|th)(?=(?:\s+[A-Za-z]+)?\s+{_FLOOR}\b)", re.IGNORECASE
)
_FLOOR_NUMBER = re.compile(rf"\b({_FLOOR}\s+)(\d{{1,2}})(?![\w]|[.,:/-]\d)", re.IGNORECASE)
_COUNT = re.compile(r"(?<![\w.,:/+–-])(\d{1,2})(?![\w]|[.,:/–-]\d)")
_LAST_WORD = re.compile(r"([^\W\d_]+)(\.?)\s*$")


def _cardinal(number: int) -> str:
    if number < 20:
        return _ONES[number]
    tens, ones = number - number % 10, number % 10
    return _TENS[tens] if not ones else f"{_TENS[tens]}-{_ONES[ones]}"


def _time(match: re.Match[str]) -> str:
    before = match.string[: match.start()].lower().split()[-3:]
    if _RATIO_WORDS & {word.strip(".,;:()") for word in before}:
        return match.group(0)
    hour, minute = int(match.group(1)), int(match.group(2))
    if minute == 0:
        return f"{_cardinal(hour)} o'clock"
    if minute < 10:
        return f"{_cardinal(hour)} oh {_cardinal(minute)}"
    return f"{_cardinal(hour)} {_cardinal(minute)}"


def _ordinal(match: re.Match[str]) -> str:
    number = int(match.group(1))
    return _ORDINALS[number] if number <= SMALL else match.group(0)


def _floor_number(match: re.Match[str]) -> str:
    number = int(match.group(2))
    if number > SMALL or _UNIT.match(match.string, match.end()):
        return match.group(0)
    return match.group(1) + _cardinal(number)


def _count(match: re.Match[str]) -> str:
    number = int(match.group(1))
    if number > SMALL or len(match.group(1)) > 1 and match.group(1)[0] == "0":
        return match.group(0)
    if _UNIT.match(match.string, match.end()):
        return match.group(0)
    last = _LAST_WORD.search(match.string[: match.start()])
    if last:
        word, abbreviated = last.group(1), last.group(2)
        if abbreviated or word.lower() in _LABELS or _STREET.search(word):
            return match.group(0)
    return _cardinal(number)


def in_words(text: str) -> str:
    """The text with its small counts, clock times and floor ordinals in words."""
    text = _TIME.sub(_time, text)
    text = _ORDINAL.sub(_ordinal, text)
    text = _FLOOR_NUMBER.sub(_floor_number, text)
    return _COUNT.sub(_count, text)
