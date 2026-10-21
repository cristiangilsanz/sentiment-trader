import pytest

from sentiment_trader.stopwords import STOPWORDS
from sentiment_trader.text import normalize, remove_stopwords, tokens_without_stopwords


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Check this https://t.co/abc123", "check this"),
        ("AAPL breaking out!!! :)", "aapl breaking out"),
        ("S&P 500 at a new high", "SP500 500 at a new high"),
        ("Buying AMZN w/ my bonus", "buying amzn with my bonus"),
        ("TSLA up 5% today", "tsla up 5 percent today"),
        ("Price target $250 for NVDA", "price target dollars 50 for nvda"),
        (
            "Above the 50EMA and 200sma",
            "above the 50 exponential moving average and 200 simple moving average",
        ),
        ("Revenue hit 3billion in Q2", "revenue hit 3 billion in second quarter"),
        ("@trader #stocks MSFT -2.5", "msft negative 2.5"),
        ("Bought 10k shares, c&h forming", "bought 10 thousand shares cup and handle forming"),
    ],
)
def test_normalizes_finance_slang(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_removes_mis_encoded_characters() -> None:
    assert normalize("great quarterâ€¦ ™") == "great quarter"


def test_stopwords_keep_market_direction_words() -> None:
    for word in ("not", "up", "down", "above", "below", "under", "over"):
        assert word not in STOPWORDS
    assert remove_stopwords("the stock is not going up") == "stock not going up"


def test_tokens_combine_both_steps() -> None:
    assert tokens_without_stopwords(["The STOCK is going UP!"]) == ["stock going up"]
