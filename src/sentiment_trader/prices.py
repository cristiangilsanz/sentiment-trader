from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def fetch_prices(tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    try:
        import yfinance
    except ImportError as error:
        raise ImportError('downloading prices needs: pip install "sentiment-trader[prices]"') from error
    prices = {}
    for ticker in tickers:
        history = yfinance.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if history.empty:
            continue
        frame = history.reset_index()[["Date", "Close"]]
        frame.columns = ["date", "close"]
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        prices[ticker] = frame
    return prices


def read_prices(path: str) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["date"])
    return {
        ticker: group[["date", "close"]].reset_index(drop=True) for ticker, group in frame.groupby("ticker")
    }
