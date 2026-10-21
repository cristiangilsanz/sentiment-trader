from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_trader.backtest import DEFAULT_CAPITAL, SeedSweep, run_backtest, sweep_seeds
from sentiment_trader.evaluation import evaluate, load_labelled
from sentiment_trader.features import daily_features, join_prices
from sentiment_trader.prices import fetch_prices, read_prices
from sentiment_trader.sentiment import MODELS, create_model

DATA = resources.files("sentiment_trader") / "data"
LABELLED_TWEETS = DATA / "labelled_tweets.csv"
NASDAQ_FEATURES = DATA / "nasdaq_2016_features.csv"
DEFAULT_SPLIT = "2016-04-15"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ImportError, ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentiment-trader",
        description="Turn the sentiment of social posts into a backtested trading strategy.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_command = commands.add_parser("evaluate", help="compare sentiment models on labelled texts")
    evaluate_command.add_argument("--data", default=str(LABELLED_TWEETS))
    evaluate_command.add_argument("--models", default="vader,naive-bayes")
    evaluate_command.set_defaults(handler=_evaluate)

    score_command = commands.add_parser("score", help="label the sentiment of every post in a CSV")
    score_command.add_argument("input")
    score_command.add_argument("output")
    score_command.add_argument("--model", default="naive-bayes", choices=sorted(MODELS))
    score_command.add_argument("--text-column", default="text")
    score_command.add_argument("--train", default=str(LABELLED_TWEETS))
    score_command.set_defaults(handler=_score)

    features_command = commands.add_parser("features", help="build daily features and join prices")
    features_command.add_argument("input")
    features_command.add_argument("output")
    features_command.add_argument("--prices", help="CSV with ticker, date and close (default: Yahoo Finance)")
    features_command.set_defaults(handler=_features)

    backtest_command = commands.add_parser(
        "backtest", help="train the strategy and compare it with the market"
    )
    backtest_command.add_argument("--features", default=str(NASDAQ_FEATURES))
    backtest_command.add_argument("--split-date", default=DEFAULT_SPLIT)
    backtest_command.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    backtest_command.add_argument("--plot", help="save the equity curves to this image")
    backtest_command.add_argument("--seed", type=int, default=1, help="random seed of the forest")
    backtest_command.add_argument("--seeds", type=int, help="repeat the backtest over this many seeds")
    backtest_command.set_defaults(handler=_backtest)
    return parser


def _evaluate(args: argparse.Namespace) -> int:
    models = [create_model(name.strip()) for name in args.models.split(",")]
    results = evaluate(models, load_labelled(args.data))
    width = max(len(result.model) for result in results)
    for result in results:
        print(f"{result.model:<{width}}  {result.accuracy:6.2f}%")
    return 0


def _score(args: argparse.Namespace) -> int:
    posts = pd.read_csv(args.input, keep_default_na=False)
    model = create_model(args.model)
    training = load_labelled(args.train)
    model.fit(training.texts, training.labels)
    posts["sentiment"] = model.predict(posts[args.text_column].astype(str).tolist())
    posts.to_csv(args.output, index=False)
    print(f"Scored {len(posts)} posts with {model.name}")
    return 0


def _features(args: argparse.Namespace) -> int:
    posts = pd.read_csv(args.input, parse_dates=["datetime"])
    features = daily_features(posts)
    if args.prices:
        prices = read_prices(args.prices)
    else:
        start = features["date"].min()
        end = features["date"].max() + pd.Timedelta(days=4)
        prices = fetch_prices(features["ticker"].unique(), start, end)
    joined = join_prices(features, prices)
    joined.to_csv(args.output, index=False)
    print(f"Built {len(joined)} daily rows for {joined['ticker'].nunique() if len(joined) else 0} tickers")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    features = pd.read_csv(args.features, parse_dates=["date"])
    if args.seeds:
        return _report_sweep(sweep_seeds(features, pd.Timestamp(args.split_date), range(1, args.seeds + 1)))
    result = run_backtest(features, pd.Timestamp(args.split_date), args.capital, args.seed)
    strategy, benchmark = result.strategy_performance, result.benchmark_performance
    print(f"{'':<16}{'Strategy':>14}{'Equal weight':>16}")
    print(f"{'Final capital':<16}{strategy.final_capital:>14,.2f}{benchmark.final_capital:>16,.2f}")
    print(f"{'Total return':<16}{strategy.total_return:>13.2f}%{benchmark.total_return:>15.2f}%")
    print(f"{'Sharpe ratio':<16}{strategy.sharpe_ratio:>14.2f}{benchmark.sharpe_ratio:>16.2f}")
    print(f"{'Max drawdown':<16}{strategy.max_drawdown:>13.2f}%{benchmark.max_drawdown:>15.2f}%")
    print(f"Hit rate {result.hit_rate * 100:.1f}%, losing picks {result.losing_pick_rate * 100:.1f}%")
    if args.plot:
        from sentiment_trader.plotting import plot_equity

        plot_equity(result).savefig(Path(args.plot), dpi=150, bbox_inches="tight")
    return 0


def _report_sweep(sweep: SeedSweep) -> int:
    returns = sweep.returns
    print(f"{'Seeds':<22}{len(returns)}")
    print(f"{'Beats equal weight':<22}{sweep.wins} / {len(returns)}")
    print(
        f"{'Median return':<22}{np.median(returns):.2f}%"
        f"  (range {returns.min():.2f}% to {returns.max():.2f}%)"
    )
    print(f"{'Median Sharpe ratio':<22}{np.median(sweep.sharpe_ratios):.2f}")
    print(f"{'Equal weight return':<22}{sweep.benchmark_return:.2f}%")
    return 0
