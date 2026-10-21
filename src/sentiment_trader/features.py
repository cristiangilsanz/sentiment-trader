from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["sentiment_weighted", "sentiment_ma", "posts", "posts_ma"]
TARGET_COLUMN = "percent_change_bin"
ENGAGEMENT_COLUMNS = ("followers", "retweets")
ROLLING_WINDOW = 3
PERCENT_CHANGE_BINS = [-100, 0, 2, 100]
LOSS, SMALL_GAIN, LARGE_GAIN = 0, 1, 2


def engagement_weights(values: pd.Series) -> pd.Series:
    mean = values.expanding().mean()
    spread = values.expanding().std().fillna(values.std())
    bonus = pd.Series(0, index=values.index)
    bonus[(values >= mean) & (values < mean + spread)] = 1
    bonus[(values >= mean + spread) & (values < mean + 2 * spread)] = 2
    bonus[values >= mean + 2 * spread] = 3
    return bonus


def daily_features(posts: pd.DataFrame) -> pd.DataFrame:
    frames = [_ticker_features(ticker, group) for ticker, group in posts.groupby("ticker", sort=True)]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["ticker", "date", *FEATURE_COLUMNS])
    )


def _ticker_features(ticker: str, posts: pd.DataFrame) -> pd.DataFrame:
    ordered = posts.sort_values("datetime").reset_index(drop=True)
    weight = pd.Series(1, index=ordered.index)
    for column in ENGAGEMENT_COLUMNS:
        if column in ordered:
            weight += engagement_weights(ordered[column].fillna(0))
    ordered["weighted"] = ordered["sentiment"] * weight
    ordered["date"] = pd.to_datetime(ordered["datetime"]).dt.normalize() + pd.Timedelta(days=1)
    daily = ordered.groupby("date").agg(sentiment_sum=("weighted", "sum"), posts=("weighted", "size"))
    daily["sentiment_weighted"] = daily["sentiment_sum"] / daily["posts"]
    daily["sentiment_ma"] = daily["sentiment_weighted"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    daily["posts_ma"] = daily["posts"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    daily = daily.reset_index()
    daily.insert(0, "ticker", ticker)
    return daily[["ticker", "date", *FEATURE_COLUMNS]]


def price_changes(prices: pd.DataFrame) -> pd.DataFrame:
    changes = prices[["date", "close"]].sort_values("date").reset_index(drop=True)
    changes["percent_change"] = (changes["close"].pct_change() * 100).shift(-1)
    changes[TARGET_COLUMN] = pd.cut(
        changes["percent_change"], PERCENT_CHANGE_BINS, labels=[LOSS, SMALL_GAIN, LARGE_GAIN]
    )
    return changes


def join_prices(features: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    joined = []
    for ticker, group in features.groupby("ticker", sort=True):
        if ticker not in prices:
            continue
        merged = group.merge(price_changes(prices[ticker]), on="date", how="left")
        joined.append(merged.dropna())
    if not joined:
        return pd.DataFrame()
    result = pd.concat(joined, ignore_index=True)
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)
    return result
