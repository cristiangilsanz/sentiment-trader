from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from sentiment_trader.text import normalize_all, tokens_without_stopwords

POSITIVE = 1
NEGATIVE = -1
DEFAULT_ALPHAS = np.arange(0.01, 10, 0.01)
DEFAULT_TRANSFORMER = "ProsusAI/finbert"


class SentimentModel(Protocol):
    name: str

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> SentimentModel: ...

    def predict(self, texts: Sequence[str]) -> np.ndarray: ...


def _as_classes(labels: Sequence[int]) -> np.ndarray:
    return (np.asarray(labels) == POSITIVE).astype(int)


def _as_labels(classes: np.ndarray) -> np.ndarray:
    return np.where(classes == 1, POSITIVE, NEGATIVE)


class VaderModel:
    name = "VADER"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> VaderModel:
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = [self._analyzer.polarity_scores(text)["compound"] for text in normalize_all(texts)]
        return np.where(np.asarray(scores) >= 0, POSITIVE, NEGATIVE)


class NaiveBayesModel:
    name = "TF-IDF + Naive Bayes"

    def __init__(self, alphas: np.ndarray = DEFAULT_ALPHAS, folds: int = 5, seed: int = 1) -> None:
        self.alphas = alphas
        self.folds = folds
        self.seed = seed
        self.best_alpha: float | None = None
        self.best_auc: float | None = None
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=False)
        self._classifier: MultinomialNB | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> NaiveBayesModel:
        features = self._vectorizer.fit_transform(tokens_without_stopwords(texts))
        classes = _as_classes(labels)
        folds = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        scores = np.array(
            [
                cross_val_score(
                    MultinomialNB(alpha=alpha), features, classes, scoring="roc_auc", cv=folds
                ).mean()
                for alpha in self.alphas
            ]
        )
        self.best_alpha = round(float(self.alphas[scores.argmax()]), 1)
        self.best_auc = float(scores.max())
        self._classifier = MultinomialNB(alpha=self.best_alpha).fit(features, classes)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("fit the model before predicting")
        features = self._vectorizer.transform(tokens_without_stopwords(texts))
        return _as_labels(self._classifier.predict_proba(features).argmax(axis=1))


class TransformerModel:
    name = "FinBERT"

    def __init__(self, model: str = DEFAULT_TRANSFORMER, batch_size: int = 32) -> None:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise ImportError(
                'the transformer model needs: pip install "sentiment-trader[transformers]"'
            ) from error
        self._classifier = pipeline("text-classification", model=model, top_k=None)
        self.batch_size = batch_size

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> TransformerModel:
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        results = self._classifier(normalize_all(texts), batch_size=self.batch_size, truncation=True)
        labels = []
        for scores in results:
            by_label = {score["label"].lower(): score["score"] for score in scores}
            positive = by_label.get("positive", 0.0)
            negative = by_label.get("negative", 0.0)
            labels.append(POSITIVE if positive >= negative else NEGATIVE)
        return np.asarray(labels)


class FineTunedBertModel:
    name = "Fine-tuned BERT"

    def __init__(
        self,
        base: str = "bert-base-uncased",
        epochs: int = 4,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        max_length: int = 48,
        seed: int = 1,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'fine-tuning BERT needs: pip install "sentiment-trader[transformers]"'
            ) from error
        torch.manual_seed(seed)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        self._model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length

    def _encode(self, texts: Sequence[str]):
        return self._tokenizer(
            tokens_without_stopwords(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> FineTunedBertModel:
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup

        torch = self._torch
        encoded = self._encode(texts)
        dataset = TensorDataset(
            encoded["input_ids"], encoded["attention_mask"], torch.tensor(_as_classes(labels))
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(loader) * self.epochs)
        self._model.train()
        for _ in range(self.epochs):
            for input_ids, attention_mask, classes in loader:
                self._model.zero_grad()
                output = self._model(
                    input_ids=input_ids.to(self._device),
                    attention_mask=attention_mask.to(self._device),
                    labels=classes.to(self._device),
                )
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        encoded = self._encode(texts)
        self._model.eval()
        classes = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                logits = self._model(
                    input_ids=encoded["input_ids"][start : start + self.batch_size].to(self._device),
                    attention_mask=encoded["attention_mask"][start : start + self.batch_size].to(
                        self._device
                    ),
                ).logits
                classes.append(logits.argmax(dim=1).cpu().numpy())
        return _as_labels(np.concatenate(classes)) if classes else np.asarray([])


MODELS = {
    "vader": VaderModel,
    "naive-bayes": NaiveBayesModel,
    "finbert": TransformerModel,
    "bert": FineTunedBertModel,
}


def create_model(name: str) -> SentimentModel:
    try:
        return MODELS[name]()
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}, choose one of {', '.join(MODELS)}") from error
