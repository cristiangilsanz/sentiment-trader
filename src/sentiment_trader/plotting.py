from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from sentiment_trader.backtest import BacktestResult

BACKGROUND = "#0d111e"
GRID = "#2a3146"
TEXT = "#c9d1e3"
STRATEGY = "#4f9dff"
BENCHMARK = "#9aa3b8"


def plot_equity(result: BacktestResult) -> Figure:
    figure, axes = plt.subplots(figsize=(12, 6), facecolor=BACKGROUND)
    axes.set_facecolor(BACKGROUND)
    curves = (
        ("Equal weight", result.benchmark, result.benchmark_performance.total_return, BENCHMARK, 2.4),
        ("Sentiment strategy", result.strategy, result.strategy_performance.total_return, STRATEGY, 3.2),
    )
    for label, equity, total_return, color, width in curves:
        axes.plot(result.dates, equity[1:], color=color, linewidth=width, label=label)
        axes.annotate(
            f"{total_return:+.2f}%",
            xy=(result.dates[-1], equity[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            color=color,
            fontsize=14,
            fontweight="bold",
            va="center",
        )
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k"))
    axes.grid(color=GRID, linewidth=0.8)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    axes.tick_params(colors=TEXT, labelsize=12)
    axes.legend(facecolor=BACKGROUND, edgecolor=GRID, labelcolor=TEXT, loc="upper left", fontsize=13)
    axes.set_title("Portfolio value", color=TEXT, fontsize=16, loc="left", pad=12)
    axes.margins(x=0.01)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
