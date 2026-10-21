import pandas as pd
import pytest

from sentiment_trader.features import (
    LARGE_GAIN,
    LOSS,
    SMALL_GAIN,
    daily_features,
    engagement_weights,
    join_prices,
    price_changes,
)


def test_engagement_rewards_posts_above_the_running_average() -> None:
    assert engagement_weights(pd.Series([10, 10, 10, 10, 1000])).tolist()[-1] == 2
    assert engagement_weights(pd.Series([10] * 20 + [1000])).tolist()[-1] == 3
    assert engagement_weights(pd.Series([100, 1])).tolist() == [1, 0]


def test_a_single_post_gets_no_bonus() -> None:
    assert engagement_weights(pd.Series([500])).tolist() == [0]


def test_daily_features_average_weighted_sentiment_for_the_next_day() -> None:
    posts = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:00", "2024-01-02 11:00"]),
            "sentiment": [1, -1, 1],
        }
    )
    features = daily_features(posts)
    assert features["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert features["posts"].tolist() == [2, 1]
    assert features["sentiment_weighted"].tolist() == [0.0, 1.0]
    assert features["sentiment_ma"].tolist() == [0.0, 0.5]
    assert features["posts_ma"].tolist() == [2.0, 1.5]


def test_daily_features_keep_tickers_apart() -> None:
    posts = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "sentiment": [1, -1],
        }
    )
    features = daily_features(posts)
    assert features.set_index("ticker")["sentiment_weighted"].to_dict() == {"AAA": 1.0, "BBB": -1.0}


def test_price_changes_look_one_day_ahead_and_bin_the_move() -> None:
    prices = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=4), "close": [100.0, 99.0, 100.0, 103.0]}
    )
    changes = price_changes(prices)
    assert changes["percent_change"].round(4).tolist()[:3] == [-1.0, 1.0101, 3.0]
    assert changes["percent_change_bin"].tolist()[:3] == [LOSS, SMALL_GAIN, LARGE_GAIN]
    assert pd.isna(changes["percent_change"].iloc[-1])


def test_join_drops_days_without_prices() -> None:
    features = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-06"]),
            "sentiment_weighted": [1.0, 1.0],
            "sentiment_ma": [1.0, 1.0],
            "posts": [1, 1],
            "posts_ma": [1.0, 1.0],
        }
    )
    prices = {
        "AAA": pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3), "close": [10.0, 11.0, 12.0]})
    }
    joined = join_prices(features, prices)
    assert joined["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert joined["percent_change"].iloc[0] == pytest.approx(100 / 11)
