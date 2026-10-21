from pathlib import Path

import pandas as pd

from sentiment_trader.cli import main


def test_backtest_prints_a_report_and_saves_the_plot(tmp_path: Path, capsys) -> None:
    plot = tmp_path / "equity.png"
    assert main(["backtest", "--plot", str(plot)]) == 0
    report = capsys.readouterr().out
    assert "Final capital" in report and "979,535.07" in report
    assert plot.stat().st_size > 0


def test_evaluate_prints_accuracy(capsys) -> None:
    assert main(["evaluate", "--models", "vader"]) == 0
    assert "VADER" in capsys.readouterr().out


def test_score_and_features_run_offline(tmp_path: Path) -> None:
    posts = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "datetime": ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00", "2024-01-04 10:00"],
            "text": ["great buy", "terrible loss", "strong rally", "awful drop"],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "date": pd.bdate_range("2024-01-01", periods=5),
            "close": [10, 11, 10, 12, 13],
        }
    )
    posts.to_csv(tmp_path / "posts.csv", index=False)
    prices.to_csv(tmp_path / "prices.csv", index=False)
    assert main(["score", str(tmp_path / "posts.csv"), str(tmp_path / "scored.csv"), "--model", "vader"]) == 0
    assert pd.read_csv(tmp_path / "scored.csv")["sentiment"].tolist() == [1, -1, 1, -1]
    arguments = ["features", str(tmp_path / "scored.csv"), str(tmp_path / "features.csv")]
    assert main([*arguments, "--prices", str(tmp_path / "prices.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "features.csv")) == 3


def test_reports_errors_without_a_traceback(capsys) -> None:
    assert main(["evaluate", "--models", "magic"]) == 1
    assert "unknown model" in capsys.readouterr().err


def test_backtest_sweeps_seeds(capsys) -> None:
    assert main(["backtest", "--seeds", "3"]) == 0
    report = capsys.readouterr().out
    assert "Beats equal weight" in report and "-2.05%" in report
