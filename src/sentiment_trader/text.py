import re
from collections.abc import Iterable

from sentiment_trader.stopwords import STOPWORDS

Rule = tuple[re.Pattern[str], str]


def _literal(old: str, new: str) -> Rule:
    return re.compile(re.escape(old)), new.replace("\\", "\\\\")


def _regex(pattern: str, new: str) -> Rule:
    return re.compile(pattern), new


_LINKS = [
    _regex(r"https?:\/\/.*[\r\n]*", " "),
    _regex(r"www.*[\r\n]*", " "),
    _literal("https", ""),
]

_PUNCTUATION = [
    _literal(": ", " "),
    _literal(", ", " "),
    _literal(". ", " "),
    _regex(r"[;\n~]", " "),
    _regex("[\\]'â€¦*™|]", ""),
    _regex(r'[\[()!?"]', ""),
    _literal("_", ""),
    _literal("w/", " with "),
    _literal("f/", " for "),
]

_EMOTICONS = [_literal(face, "") for face in (":)", ":-)", ":(", ":-(", "0_o", ";)", "=^.^=")]

_AMPERSANDS = [
    _literal("%", " percent "),
    _literal(" & ", " and "),
    _literal("&amp", " and "),
    _literal("&gt", " greater than "),
    _literal("cup&handle", "cup and handle"),
    _literal("c&h", "cup and handle"),
    _literal("head&shoulders", "head and shoulders"),
    _literal("h&s", "head and shoulders"),
    _literal("point&figure", "point and figure"),
    _literal("p&f", "point and figure"),
    _literal("s&p", "SP500"),
    _literal("q&a", "question and answer"),
    _literal("&", " and "),
]

_MENTIONS = [
    _regex(r"@[a-z0-9]+", ""),
    _regex(r"#[a-z0-9]+", ""),
    _literal("@", ""),
    _literal("#", ""),
]

_COMMAS_AND_PERIODS = [
    _regex(r"([a-z]),([a-z])", r"\1 \2"),
    _regex(r"([0-9]),([0-9])", r"\1\2"),
    _regex(r"([0-9])[+]+", r"\1 "),
    _literal(",", ""),
    _literal("u.s.", " us "),
    _regex(r"\.{2,}", " "),
    _regex(r"([a-z])\.([a-z])", r"\1 \2"),
    _literal("pdating", "updating"),
    _regex(r"([a-z])\.", r"\1 "),
    _regex(r"\.([a-z])", r" \1"),
    _literal(" . ", " "),
]

_PLUS_SIGNS = [
    _regex(r"[+]([0-9])", r"positive \1"),
    _literal("c+h", "cup and handle"),
    _literal("h+s", "head and shoulders"),
    _literal("cup+handle", "cup and handle"),
    _literal(" + ", " and "),
    _literal("+ ", " "),
    _regex(r"([a-z])[+]([a-z])", r"\1 and \2"),
    _literal("+", ""),
]

_MINUS_SIGNS = [
    _regex(r"([a-z])[-]+([a-z])", r"\1 \2"),
    _regex(r"([a-z]) - ([a-z])", r"\1 to \2"),
    _regex(r"([0-9]) -([0-9\.])", r"\1 to \2"),
    _regex(r" [-]([0-9])", r" negative \1"),
    _regex(r"([0-9])-([0-9\.])", r"\1 to \2"),
    _regex(r"([0-9]) - ([0-9\.])", r"\1 to \2"),
    _regex(r"([0-9a-z])-([0-9a-z])", r"\1 \2"),
    _regex(r"[-]+[>]", " "),
    _regex(r" [-]+ ", " "),
    _literal("-", " "),
]

_SYMBOLS = [
    _regex(r"[$][0-9\.]", " dollars "),
    _literal("$", ""),
    _literal("=", " equals "),
    _literal("b/c", " because "),
    _literal("b/out", " break out "),
    _literal("b/o", " break out "),
    _literal("p/e", " pe ratio "),
    _regex(r" [/]+ ", " "),
    _literal(" 1/2 ", " .5 "),
    _literal(" 1/4 ", " .25 "),
    _literal(" 3/4 ", " .75 "),
    _literal(" 1/3 ", " .3 "),
    _literal(" 2/3 ", " .6 "),
    _regex(r"[/]{2,}", " "),
    _regex(r"([a-z])/([a-z])", r"\1 and \2"),
    _regex(r"[0-9]+/[0-9]+/[0-9]+", ""),
    _regex(r"([0-9]{3,})/([0-9\.]{2,})", r"\1 to \2"),
    _regex(r"([0-9]{2,})/([0-9\.]{3,})", r"\1 to \2"),
    _regex(r"[a-z0-9]+/[a-z0-9]+", " "),
    _literal("/", ""),
    _regex(r"[<]+ ", " "),
    _literal("<", " less than "),
    _regex(r" [>]+", " "),
    _literal(">", " greater than "),
    _regex(r"[0-9]+:[0-9]+am", " "),
    _regex(r"[0-9]+:[0-9]", " "),
    _literal(":", " "),
    _literal("user ", " "),
]

_MOVING_AVERAGES = [
    rule
    for short, name in (
        ("dma", "displaced moving average"),
        ("sma", "simple moving average"),
        ("ema", "exponential moving average"),
        ("ma", "moving average"),
    )
    for rule in (
        _regex(rf"([0-9]+){short}", rf"\1 {name} "),
        _regex(rf"{short}([0-9]+)", rf"\1 {name} "),
    )
]

_TIME_UNITS = [
    _regex(r"([0-9])mos", r"\1 months "),
    _regex(r"([0-9])minute", r"\1 minute "),
    _regex(r"([0-9])minutes", r"\1 minutes "),
    _regex(r"([0-9])min", r"\1 minute "),
    _regex(r"([0-9])mins", r"\1 minutes "),
    _regex(r"([0-9])day", r"\1 day "),
    _regex(r"([0-9])days", r"\1 days "),
    _regex(r"([0-9])wk", r"\1 week "),
    _literal(" wk ", " week "),
    _literal(" wknd ", " weekend "),
    _regex(r"([0-9])wks", r"\1 weeks "),
    _regex(r"([0-9])hours", r"\1 hours "),
    _regex(r"([0-9])hour", r"\1 hour "),
    _regex(r"([0-9])yr", r"\1 year "),
    _regex(r"([0-9])yrs", r"\1 years "),
    _literal(" yr", " year "),
    _regex(r"([0-9])am", r"\1 am "),
    _regex(r"([0-9])pm", r"\1 pm "),
    _regex(r"([0-9])est", r"\1 "),
    _regex(r"([0-9])ish", r"\1 "),
    _regex(r"([0-9 ])pts", r"\1 points "),
    _regex(r"([0-9])x", r"\1 times "),
    *(_regex(rf"([0-9]){suffix}", r"\1 ") for suffix in ("th", "rd", "st", "nd")),
]

_SLANG = [
    _literal("mrkt", "market"),
    _literal(" vol ", " volume "),
    _literal(" ptrend", " positive trend "),
    _literal(" ppl", " people "),
    _literal(" pts", " points "),
    _literal(" pt", " point "),
    _regex(r" l(ol){1,}", " laugh "),
    _literal("imho", " in my opinion "),
    _literal("prev ", "previous "),
    *(
        _literal(f" {code}", f" {name} quarter ")
        for number, name in ((1, "first"), (2, "second"), (3, "third"), (4, "fourth"))
        for code in (f"{number}q", f"q{number}")
    ),
    _literal(" 10q ", " form 10 "),
]

_QUANTITIES = [
    _regex(r"([0-9])million", r"\1 million "),
    _regex(r"([0-9])mil", r"\1 million "),
    _literal(" mil ", " million "),
    _regex(r"([0-9])billion", r"\1 billion "),
    _regex(r"([0-9])cents", r"\1 cents "),
    _regex(r"([0-9])3d", r"\1 3 dimensional "),
    _regex(r"([0-9])gb", r"\1 gigabytes "),
    _regex(r"([0-9])c", r"\1 calls "),
    _regex(r"([0-9])y", r"\1 year "),
    _regex(r"([0-9])p", r"\1 puts "),
    _regex(r"([0-9])d", r"\1 days "),
    _regex(r"([0-9])h", r"\1 hour "),
    _regex(r"([0-9])s", r"\1 "),
    _regex(r"([0-9])k1", r"\1 thousand "),
    _regex(r"([0-9])k", r"\1 thousand "),
    _regex(r"([0-9])m", r"\1 million "),
    _regex(r"([0-9])b", r"\1 billion "),
    _regex(r"([0-9])([a-z])", r"\1 \2"),
    _regex(r" +", " "),
]

RULES: list[Rule] = [
    *_LINKS,
    *_PUNCTUATION,
    *_EMOTICONS,
    *_AMPERSANDS,
    *_MENTIONS,
    *_COMMAS_AND_PERIODS,
    *_PLUS_SIGNS,
    *_MINUS_SIGNS,
    *_SYMBOLS,
    *_MOVING_AVERAGES,
    *_TIME_UNITS,
    *_SLANG,
    *_QUANTITIES,
]

_EDGE_CHARACTERS = " .!?,)(:-"


def normalize(text: str) -> str:
    cleaned = text.lower()
    for pattern, replacement in RULES:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip(_EDGE_CHARACTERS)


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in STOPWORDS)


def normalize_all(texts: Iterable[str]) -> list[str]:
    return [normalize(text) for text in texts]


def tokens_without_stopwords(texts: Iterable[str]) -> list[str]:
    return [remove_stopwords(normalize(text)) for text in texts]
