from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_trader.sentiment import SentimentModel


@dataclass(frozen=True)
class LabelledTexts:
    texts: list[str]
    labels: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    model: str
    accuracy: float


def load_labelled(
    path: str | Path, text_column: str = "text", label_column: str = "sentiment"
) -> LabelledTexts:
    frame = pd.read_csv(path, keep_default_na=False)
    return LabelledTexts(frame[text_column].astype(str).tolist(), frame[label_column].to_numpy())


def split_indices(count: int, train_fraction: float = 0.8, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.random.RandomState(seed).permutation(count)
    cut = int(train_fraction * count)
    return shuffled[:cut], shuffled[cut:]


def evaluate(
    models: Iterable[SentimentModel],
    data: LabelledTexts,
    train_fraction: float = 0.8,
    seed: int = 1,
) -> list[Evaluation]:
    train, test = split_indices(len(data.texts), train_fraction, seed)
    train_texts = [data.texts[i] for i in train]
    test_texts = [data.texts[i] for i in test]
    results = []
    for model in models:
        model.fit(train_texts, data.labels[train])
        predictions = model.predict(test_texts)
        accuracy = float((predictions == data.labels[test]).mean() * 100)
        results.append(Evaluation(model.name, accuracy))
    return results
