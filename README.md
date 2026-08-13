# Text Mining for Stress Analysis Using Natural Language Processing (NLP) Techniques

![CI](https://github.com/SmashCodeJJ/stress-analysis-nlp/actions/workflows/ci.yml/badge.svg)

**Author:** Youxin Zhuo ([SmashCodeJJ](https://github.com/SmashCodeJJ))  
**Institution:** Penn State University  
**Domain:** Natural Language Processing · Machine Learning · Mental Health Text Classification

---

## Overview

This project builds a machine learning pipeline to classify text as **stress** or **non-stress** using NLP preprocessing and multiple classification algorithms. The goal is to compare model performance and identify the best approach for automated stress detection from social media text.

## Problem Statement

Mental health stress detection from text is a valuable NLP application. This project implements a full pipeline — from raw text to trained classifiers — and benchmarks four algorithms on a labeled Reddit stress dataset.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Dreaddit Dataset (Kaggle)](https://www.kaggle.com/datasets/menekse/stress-analysis-for-social-media) |
| Samples | ~3,000+ labeled posts |
| Features | Text content + metadata (subreddit, label, etc.) |
| Labels | `stress` (1) vs `no stress` (0) |

> **Note:** Download `dreaddit.csv` from Kaggle and place it in the project root before running the notebook.

---

## Methodology

### 1. Text Preprocessing

Combined **spaCy** and **NLTK** for comprehensive NLP preprocessing:

| Step | Library | Description |
|------|---------|-------------|
| Lemmatization | spaCy | Reduce words to base form |
| Stopword removal | spaCy | Remove common non-informative words |
| Stemming | NLTK (Snowball) | Further normalize word forms |

spaCy simplifies lemmatization and stopword removal; NLTK provides stemming (not available in spaCy).

### 2. Feature Extraction

- **TF-IDF Vectorizer** with bi-grams (`ngram_range=(1,2)`) — weights terms by importance across documents
- **CountVectorizer** with tri-grams for tree-based models
- `min_df=2`, `max_df=0.85`, `sublinear_tf=True` to reduce noise from rare/common terms
- 80/20 train-test split via `sklearn.model_selection.train_test_split`

### 3. Model Pipeline

Each classifier is wrapped in a scikit-learn `Pipeline`:

```
TfidfVectorizer / CountVectorizer → Classifier
```

Hyperparameter tuning via **GridSearchCV** is applied to the Random Forest model (tri-gram features). ComplementNB uses tuned TF-IDF settings (`max_df=0.85`, `alpha=1.0`) from cross-validation.

### 4. Classifiers Evaluated

| Algorithm | Description |
|-----------|-------------|
| **ComplementNB** | Naive Bayes variant suited for imbalanced text |
| **MultinomialNB** | Classic Naive Bayes for word counts |
| **BernoulliNB** | Binary word presence features |
| **Random Forest** | Ensemble with tri-gram CountVectorizer |
| **LinearSVC** | Fast linear SVM with balanced class weights |
| **Logistic Regression** | Linear classifier with balanced class weights |

---

## Results

Model comparison on the held-out test set (711 samples, verified run):

| Model | Accuracy | Stress F1 | Stress Recall | Notes |
|-------|----------|-----------|---------------|-------|
| **ComplementNB + TF-IDF (tuned)** | **73%** | **0.77** | **85%** | **Best model** — tuned `max_df=0.85`, `alpha=1.0` |
| Random Forest + Count (1,3) | 73% | 0.76 | 82% | Tri-gram bag-of-words |
| Random Forest + Count (1,3) [tuned] | 72% | 0.75 | 82% | GridSearchCV optimized |
| LinearSVC + TF-IDF (1,2) | 72% | 0.74 | 76% | Fast linear classifier |
| Logistic Regression + TF-IDF (1,2) | 72% | 0.74 | 76% | Balanced class weights |
| MultinomialNB + TF-IDF (1,2) | 71% | 0.74 | 78% | Strong stress recall |
| ComplementNB + TF-IDF (1,2) | 71% | 0.73 | 76% | Default hyperparameters |
| BernoulliNB + Binary Count (1,2) | 71% | 0.73 | 74% | Binary word presence |

**Best model (verified):** ComplementNB + TF-IDF (tuned) — **73% accuracy**, stress F1=**0.77**, stress recall=**85%**.

Improvements over the original baseline (single CountVectorizer, no n-grams):
- Tuned ComplementNB with TF-IDF bi-grams (+4% stress F1 vs original Random Forest baseline)
- Tri-gram features for Random Forest (+3% accuracy vs basic pipeline)
- GridSearchCV hyperparameter tuning for Random Forest
- Preprocessing cache for reproducible CI runs

See [`results/model_comparison.json`](results/model_comparison.json) for full classification reports and sample predictions.

### Classification Report (Best Model: ComplementNB + TF-IDF tuned)

```
              precision    recall  f1-score   support

   no stress       0.79      0.60      0.68       339
      stress       0.70      0.85      0.77       372

    accuracy                           0.73       711
```

---

## Project Structure

```
stress-analysis-nlp/
├── README.md
├── run_stress_analysis.py      # Improved reproducible pipeline
├── stress_detection.ipynb      # Original analysis notebook
├── requirements.txt
├── scripts/
│   └── download_dataset.py
├── results/
│   ├── model_comparison.json   # Verified metrics + sample predictions
│   └── preprocessed_cache.csv  # Cached spaCy preprocessing (CI speedup)
└── dreaddit.csv                # Download via script (not in git)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SmashCodeJJ/stress-analysis-nlp.git
cd stress-analysis-nlp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download the dataset

Download `dreaddit.csv` from [Kaggle](https://www.kaggle.com/datasets/menekse/stress-analysis-for-social-media) and place it in the project root.

### 4. Run the notebook

```bash
jupyter notebook stress_detection.ipynb
```

### 5. Run the full pipeline (reproducible script)

```bash
python scripts/download_dataset.py   # downloads dreaddit.csv if missing
python run_stress_analysis.py        # trains 6 models + GridSearch, saves results/
```

---

## Key Libraries

- **spaCy** — NLP preprocessing (lemmatization, stopwords)
- **NLTK** — Stemming (Snowball Stemmer)
- **scikit-learn** — TF-IDF, CountVectorizer, Pipeline, GridSearchCV, classifiers
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization

---

## Key Learnings

- **Tuned ComplementNB wins on F1:** TF-IDF bi-grams with `max_df=0.85` and `alpha=1.0` reach stress F1=0.77 — highest among all configs tested.
- **Tri-grams help tree models:** Random Forest with CountVectorizer(1,3) matches accuracy but slightly lower F1.
- **TF-IDF improves linear models:** Bi-gram TF-IDF features boost LinearSVC and Logistic Regression over raw counts.
- **High stress recall:** The best model catches 85% of stress posts — useful when missing stress cases is costly.

---

## Future Work

- Word embeddings (Word2Vec, GloVe) or transformer models (BERT)
- Deploy as a web demo with Gradio or Streamlit
- Error analysis on misclassified posts (e.g., sarcasm, mixed sentiment)

---

## License

Academic project — Penn State University.
