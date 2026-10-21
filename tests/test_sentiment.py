import numpy as np
import pytest

from sentiment_trader.evaluation import evaluate, load_labelled, split_indices
from sentiment_trader.sentiment import NEGATIVE, POSITIVE, NaiveBayesModel, VaderModel, create_model
from tests.conftest import LABELLED_TWEETS


def test_vader_reads_obvious_sentiment() -> None:
    predictions = VaderModel().predict(["Great earnings, I love this stock", "Terrible guidance, awful loss"])
    assert predictions.tolist() == [POSITIVE, NEGATIVE]


def test_naive_bayes_learns_from_examples() -> None:
    texts = [
        "stock going up strong buy",
        "bullish breakout up big",
        "stock crashing sell now",
        "bearish breakdown sell",
    ] * 5
    labels = [POSITIVE, POSITIVE, NEGATIVE, NEGATIVE] * 5
    model = NaiveBayesModel(alphas=np.array([0.5, 1.0])).fit(texts, labels)
    assert model.predict(["strong breakout up", "sell this crashing stock"]).tolist() == [POSITIVE, NEGATIVE]
    assert model.best_alpha in (0.5, 1.0)


def test_naive_bayes_needs_training_first() -> None:
    with pytest.raises(RuntimeError):
        NaiveBayesModel().predict(["anything"])


def test_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        create_model("magic")


def test_split_is_deterministic_and_disjoint() -> None:
    train, test = split_indices(100)
    again, _ = split_indices(100)
    assert len(train) == 80 and len(test) == 20
    assert set(train).isdisjoint(test)
    assert train.tolist() == again.tolist()


def test_reproduces_the_original_naive_bayes_accuracy() -> None:
    model = NaiveBayesModel()
    [result] = evaluate([model], load_labelled(LABELLED_TWEETS))
    assert result.accuracy == pytest.approx(68.68, abs=0.01)
    assert model.best_alpha == 0.7
    assert model.best_auc == pytest.approx(0.8317, abs=0.0001)
