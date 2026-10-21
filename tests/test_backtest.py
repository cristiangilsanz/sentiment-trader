import platform

import numpy as np
import pandas as pd
import pytest

from sentiment_trader.backtest import performance, run_backtest, sweep_seeds
from tests.conftest import NASDAQ_FEATURES


def test_performance_metrics() -> None:
    result = performance(np.array([100.0, 110.0, 99.0, 121.0]))
    assert result.total_return == pytest.approx(21.0)
    assert result.max_drawdown == pytest.approx(-10.0)
    assert result.sharpe_ratio > 0


def test_flat_equity_has_no_sharpe_ratio() -> None:
    assert performance(np.array([100.0, 100.0, 100.0])).sharpe_ratio == 0.0


def test_split_date_must_leave_data_on_both_sides() -> None:
    features = pd.read_csv(NASDAQ_FEATURES, parse_dates=["date"])
    with pytest.raises(ValueError):
        run_backtest(features, pd.Timestamp("2030-01-01"))


ARM = platform.machine().lower() in {"arm64", "aarch64"}


@pytest.mark.skipif(ARM, reason="the forest's floating-point results differ on ARM")
def test_reproduces_the_original_backtest() -> None:
    features = pd.read_csv(NASDAQ_FEATURES, parse_dates=["date"])
    result = run_backtest(features, pd.Timestamp("2016-04-15"))
    assert result.strategy_performance.final_capital == pytest.approx(1006028.63, abs=0.01)
    assert result.benchmark_performance.final_capital == pytest.approx(979535.07, abs=0.01)
    assert result.losing_pick_rate == pytest.approx(0.2010050251, abs=1e-9)
    assert len(result.dates) == 44


def test_the_strategy_beats_the_market_on_most_seeds() -> None:
    features = pd.read_csv(NASDAQ_FEATURES, parse_dates=["date"])
    sweep = sweep_seeds(features, pd.Timestamp("2016-04-15"), range(1, 11))
    assert sweep.benchmark_return == pytest.approx(-2.05, abs=0.01)
    assert sweep.wins >= 8
