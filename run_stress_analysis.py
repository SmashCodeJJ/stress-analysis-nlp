#!/usr/bin/env python3
"""Run the stress analysis pipeline and save results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dreaddit.csv"
RESULTS_PATH = ROOT / "results" / "model_comparison.json"


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
        tokens = []
        for token in nlp(text):
            if token.is_stop or token.is_punct:
                continue
            tokens.append(stemmer.stem(token.lemma_))
        return " ".join(tokens)

    return preprocess


def parse_report(report_text: str) -> dict:
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    metrics = {}
    for line in lines:
        if line.startswith("accuracy"):
            metrics["accuracy"] = float(line.split()[-2])
        elif line.startswith("stress"):
            parts = line.split()
            metrics["stress_precision"] = float(parts[1])
            metrics["stress_recall"] = float(parts[2])
            metrics["stress_f1"] = float(parts[3])
    return metrics


def evaluate_model(name: str, estimator, x_train, x_test, y_train, y_test) -> dict:
    clf = Pipeline([
        ("vectorizer_bow", CountVectorizer()),
        (name, estimator),
    ])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(report)
    parsed = parse_report(report)
    parsed["model"] = name
    parsed["report"] = report
    return parsed


def main() -> int:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(df["detection"].value_counts())

    print("Preprocessing text (spaCy + NLTK)...")
    preprocess = build_preprocessor()
    df["preprocessed_txt"] = df["text"].apply(preprocess)

    x_train, x_test, y_train, y_test = train_test_split(
        df["preprocessed_txt"],
        df["detection"],
        test_size=0.2,
        random_state=2022,
        stratify=df["detection"],
    )
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    results = [
        evaluate_model("MultinomialNB", MultinomialNB(), x_train, x_test, y_train, y_test),
        evaluate_model("SVM", SVC(), x_train, x_test, y_train, y_test),
        evaluate_model("KNN", KNeighborsClassifier(n_neighbors=10, metric="euclidean"), x_train, x_test, y_train, y_test),
        evaluate_model("RandomForest", RandomForestClassifier(random_state=2022), x_train, x_test, y_train, y_test),
    ]

    best = max(results, key=lambda item: item.get("stress_f1", 0))
    print(f"\nBest model by stress F1: {best['model']} (F1={best['stress_f1']:.2f}, accuracy={best['accuracy']:.2f})")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"models": results, "best_model": best["model"]}, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
