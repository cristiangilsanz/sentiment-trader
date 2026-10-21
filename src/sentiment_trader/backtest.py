from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sentiment_trader.features import FEATURE_COLUMNS, LOSS, TARGET_COLUMN

TRADING_DAYS_PER_YEAR = 252
DEFAULT_CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class Performance:
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass(frozen=True)
class BacktestResult:
    dates: pd.DatetimeIndex
    strategy: np.ndarray
    benchmark: np.ndarray
    losing_pick_rate: float
    hit_rate: float

    @property
    def strategy_performance(self) -> Performance:
        return performance(self.strategy)

    @property
    def benchmark_performance(self) -> Performance:
        return performance(self.benchmark)


def performance(equity: np.ndarray) -> Performance:
    returns = np.diff(equity) / equity[:-1]
    spread = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / spread * np.sqrt(TRADING_DAYS_PER_YEAR)) if spread > 0 else 0.0
    peaks = np.maximum.accumulate(equity)
    return Performance(
        final_capital=float(equity[-1]),
        total_return=float((equity[-1] - equity[0]) / equity[0] * 100),
        sharpe_ratio=sharpe,
        max_drawdown=float(((equity - peaks) / peaks).min() * 100),
    )


def split_by_date(features: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = features.sort_values(["date", "ticker"])
    train = ordered[ordered["date"] < split_date].reset_index(drop=True)
    test = ordered[ordered["date"] >= split_date].reset_index(drop=True)
    return train, test


def run_backtest(
    features: pd.DataFrame,
    split_date: pd.Timestamp,
    initial_capital: float = DEFAULT_CAPITAL,
    seed: int = 1,
) -> BacktestResult:
    train, test = split_by_date(features, split_date)
    if train.empty or test.empty:
        raise ValueError("the split date must leave data on both sides")
    model = RandomForestClassifier(random_state=seed)
    model.fit(train[FEATURE_COLUMNS].to_numpy(), train[TARGET_COLUMN].to_numpy())
    test = test.assign(prediction=model.predict(test[FEATURE_COLUMNS].to_numpy()))

    buys = test["prediction"] > LOSS
    losing_pick_rate = float((buys & (test[TARGET_COLUMN] == LOSS)).sum() / len(test))
    hit_rate = float((test.loc[buys, TARGET_COLUMN] > LOSS).mean()) if buys.any() else 0.0

    strategy = [initial_capital]
    benchmark = [initial_capital]
    dates = sorted(test["date"].unique())
    for date in dates:
        day = test[test["date"] == date]
        benchmark.append(benchmark[-1] + _equal_weight_profit(benchmark[-1], day["percent_change"]))
        picks = day.loc[day["prediction"] > LOSS, "percent_change"]
        strategy.append(strategy[-1] + _equal_weight_profit(strategy[-1], picks))
    return BacktestResult(
        dates=pd.DatetimeIndex(dates),
        strategy=np.asarray(strategy),
        benchmark=np.asarray(benchmark),
        losing_pick_rate=losing_pick_rate,
        hit_rate=hit_rate,
    )


@dataclass(frozen=True)
class SeedSweep:
    returns: np.ndarray
    sharpe_ratios: np.ndarray
    benchmark_return: float

    @property
    def wins(self) -> int:
        return int((self.returns > self.benchmark_return).sum())


def sweep_seeds(
    features: pd.DataFrame,
    split_date: pd.Timestamp,
    seeds: range,
    initial_capital: float = DEFAULT_CAPITAL,
) -> SeedSweep:
    results = [run_backtest(features, split_date, initial_capital, seed) for seed in seeds]
    return SeedSweep(
        returns=np.array([result.strategy_performance.total_return for result in results]),
        sharpe_ratios=np.array([result.strategy_performance.sharpe_ratio for result in results]),
        benchmark_return=results[0].benchmark_performance.total_return,
    )


def _equal_weight_profit(capital: float, percent_changes: pd.Series) -> float:
    if percent_changes.empty:
        return 0.0
    return float(((capital / len(percent_changes)) * (percent_changes / 100)).sum())
