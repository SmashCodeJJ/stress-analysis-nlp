#!/usr/bin/env python3
"""Run the improved stress analysis pipeline and save results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dreaddit.csv"
CACHE_PATH = ROOT / "results" / "preprocessed_cache.csv"
RESULTS_PATH = ROOT / "results" / "model_comparison.json"

TFIDF_KWARGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
}
COUNT_BIGRAM_KWARGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "binary": True,
}
COUNT_TRIGRAM_KWARGS = {
    "ngram_range": (1, 3),
    "min_df": 2,
    "max_df": 0.85,
}

SAMPLE_TEXTS = [
    "I am feeling very stressed lately because of my workload at work. I have been working long hours and feeling overwhelmed.",
    "The weather outside is lovely today. The birds are chirping, and the trees are swaying in the breeze.",
    "The sun is shining bright today, and the birds are singing happily in the trees. I feel calm and relaxed.",
    "We'd be saving so much money with this.",
    "Despite the stress of daily life, I always make time for things that bring me joy.",
    "My heart is racing and I can't seem to catch my breath.",
    "I can't seem to sleep at night because I'm so anxious.",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download dreaddit.csv or run: python scripts/download_dataset.py"
        )
    messages = pd.read_csv(path)
    detection = {0: "no stress", 1: "stress"}
    messages["detection"] = messages["label"].map(detection)
    return messages.loc[:, ["text", "label", "detection"]]


def build_preprocessor():
    nlp = spacy.load("en_core_web_sm")
    stemmer = SnowballStemmer("english")
    stopwords = nlp.Defaults.stop_words
    for word in ["no", "not", "n't"]:
        stopwords.discard(word)
    nlp.Defaults.stop_words = stopwords

    def preprocess(text: str) -> str:
        if pd.isna(text):
            return ""
        tokens = []
        for token in nlp(str(text)):
            if token.is_stop or token.is_punct:
                continue
            tokens.append(stemmer.stem(token.lemma_))
        return " ".join(tokens)

    return preprocess


def load_or_build_preprocessed(df: pd.DataFrame) -> pd.Series:
    if CACHE_PATH.exists():
        print(f"Loading cached preprocessing from {CACHE_PATH.name}...")
        cached = pd.read_csv(CACHE_PATH)
        if len(cached) == len(df):
            return cached["preprocessed_txt"].fillna("")

    print("Preprocessing text (spaCy + NLTK)...")
    preprocess = build_preprocessor()
    processed = df["text"].apply(preprocess)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_frame(name="preprocessed_txt").to_csv(CACHE_PATH, index=False)
    print(f"Saved preprocessing cache to {CACHE_PATH}")
    return processed


def parse_report(report_text: str) -> dict:
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    metrics: dict = {}
    for line in lines:
        if line.startswith("accuracy"):
            metrics["accuracy"] = float(line.split()[-2])
        elif line.startswith("stress"):
            parts = line.split()
            metrics["stress_precision"] = float(parts[1])
            metrics["stress_recall"] = float(parts[2])
            metrics["stress_f1"] = float(parts[3])
    return metrics


def evaluate_pipeline(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(report)
    parsed = parse_report(report)
    parsed["model"] = name
    parsed["report"] = report
    return parsed, pipeline


def build_model_configs() -> list[tuple[str, Pipeline]]:
    return [
        (
            "ComplementNB + Tfidf(1,2)",
            Pipeline([
                ("vectorizer", TfidfVectorizer(**TFIDF_KWARGS)),
                ("classifier", ComplementNB(alpha=0.1)),
            ]),
        ),
        (
            "MultinomialNB + Tfidf(1,2)",
            Pipeline([
                ("vectorizer", TfidfVectorizer(**TFIDF_KWARGS)),
                ("classifier", MultinomialNB(alpha=0.05)),
            ]),
        ),
        (
            "BernoulliNB + Count(1,2) binary",
            Pipeline([
                ("vectorizer", CountVectorizer(**COUNT_BIGRAM_KWARGS)),
                ("classifier", BernoulliNB(alpha=0.1)),
            ]),
        ),
        (
            "RandomForest + Count(1,3)",
            Pipeline([
                ("vectorizer", CountVectorizer(**COUNT_TRIGRAM_KWARGS)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=60,
                        min_samples_leaf=1,
                        random_state=2022,
                        n_jobs=-1,
                    ),
                ),
            ]),
        ),
        (
            "LinearSVC + Tfidf(1,2)",
            Pipeline([
                ("vectorizer", TfidfVectorizer(**TFIDF_KWARGS)),
                (
                    "classifier",
                    LinearSVC(C=0.5, class_weight="balanced", random_state=2022, max_iter=3000),
                ),
            ]),
        ),
        (
            "LogisticRegression + Tfidf(1,2)",
            Pipeline([
                ("vectorizer", TfidfVectorizer(**TFIDF_KWARGS)),
                (
                    "classifier",
                    LogisticRegression(
                        C=3.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=2022,
                    ),
                ),
            ]),
        ),
    ]


def tune_random_forest(x_train, y_train) -> Pipeline:
    print("\n=== GridSearch: RandomForest + Count(1,3) ===")
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(**COUNT_TRIGRAM_KWARGS)),
        ("classifier", RandomForestClassifier(random_state=2022, n_jobs=-1)),
    ])
    grid = {
        "classifier__n_estimators": [400, 500],
        "classifier__max_depth": [50, 60, None],
        "classifier__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=make_scorer(f1_score, pos_label="stress"),
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV stress F1: {search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


def main() -> int:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df["detection"].value_counts())

    df["preprocessed_txt"] = load_or_build_preprocessed(df)

    x_train, x_test, y_train, y_test = train_test_split(
        df["preprocessed_txt"],
        df["detection"],
        test_size=0.2,
        random_state=2022,
        stratify=df["detection"],
    )
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in build_model_configs():
        metrics, fitted = evaluate_pipeline(name, pipeline, x_train, x_test, y_train, y_test)
        results.append(metrics)
        fitted_models[name] = fitted

    tuned_rf, tuned_params = tune_random_forest(x_train, y_train)
    y_pred = tuned_rf.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(report)
    tuned_metrics = parse_report(report)
    tuned_metrics["model"] = "RandomForest + Count(1,3) [tuned]"
    tuned_metrics["report"] = report
    tuned_metrics["best_params"] = tuned_params
    results.append(tuned_metrics)
    fitted_models[tuned_metrics["model"]] = tuned_rf

    best = max(results, key=lambda item: (item.get("stress_f1", 0), item.get("accuracy", 0)))
    best_pipeline = fitted_models[best["model"]]

    print(
        f"\nBest model: {best['model']} "
        f"(F1={best['stress_f1']:.3f}, accuracy={best['accuracy']:.3f})"
    )

    sample_predictions = best_pipeline.predict(SAMPLE_TEXTS)
    print("\nSample predictions:")
    for text, label in zip(SAMPLE_TEXTS, sample_predictions):
        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"  [{label}] {preview}")

    output = {
        "methodology": {
            "preprocessing": "spaCy lemmatization + stopword removal + NLTK Snowball stemming",
            "negation_handling": "Keeps 'no', 'not', \"n't\" in vocabulary",
            "feature_extraction": "TF-IDF bi-grams or CountVectorizer bi/tri-grams",
            "train_test_split": "80/20 stratified, random_state=2022",
        },
        "models": results,
        "best_model": best["model"],
        "best_metrics": {
            "accuracy": best["accuracy"],
            "stress_f1": best["stress_f1"],
            "stress_precision": best.get("stress_precision"),
            "stress_recall": best.get("stress_recall"),
        },
        "sample_predictions": [
            {"text": text, "prediction": label}
            for text, label in zip(SAMPLE_TEXTS, sample_predictions)
        ],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
