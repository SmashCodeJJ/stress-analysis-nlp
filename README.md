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
CountVectorizer → Classifier
```

### 4. Classifiers Evaluated

| Algorithm | Description |
|-----------|-------------|
| **Multinomial Naive Bayes** | Probabilistic classifier suited for text frequency features |
| **Support Vector Machine (SVM)** | Maximum-margin classifier |
| **K-Nearest Neighbors (KNN)** | Instance-based learning (k=10, euclidean) |
| **Random Forest** | Ensemble of decision trees |

---

## Results

Model comparison on the held-out test set:

| Model | Accuracy | Stress F1 | Notes |
|-------|----------|-----------|-------|
| **ComplementNB + TF-IDF (tuned)** | **73%** | **0.77** | Best model — bi-grams + tuned alpha (verified run) |
| Random Forest + Count (1,3) | 72% | 0.76 | Strong recall on stress class |
| MultinomialNB + TF-IDF | 71% | 0.75 | Good baseline with TF-IDF |
| Logistic Regression + TF-IDF | 73% | 0.76 | Balanced performance |
| LinearSVC + TF-IDF | 72% | 0.74 | Fast linear classifier |
| BernoulliNB + Binary Count | 71% | 0.73 | Binary word presence features |
| MultinomialNB (baseline) | 71% | 0.75 | Original CountVectorizer |
| SVM (baseline) | 70% | 0.72 | Original notebook config |
| KNN (baseline) | 58% | 0.46 | Poor on sparse text features |

**Best model (verified):** ComplementNB + TF-IDF — 73% accuracy, stress F1=**0.77** (improved from 0.76 via bi-gram TF-IDF and hyperparameter tuning).

See [`results/model_comparison.json`](results/model_comparison.json) for full classification reports.

### Classification Report (Multinomial Naive Bayes)

```
              precision    recall  f1-score   support

   no stress       0.76      0.60      0.67       339
      stress       0.70      0.83      0.76       372

    accuracy                           0.72       711
```

---

## Project Structure

```
stress-analysis-nlp/
├── README.md                 # This file
├── stress_detection.ipynb    # Full analysis notebook
├── requirements.txt          # Python dependencies
└── dreaddit.csv              # Dataset (download from Kaggle — not included)
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
python run_stress_analysis.py        # trains all 4 models, saves results/
```

---

## Key Libraries

- **spaCy** — NLP preprocessing (lemmatization, stopwords)
- **NLTK** — Stemming (Snowball Stemmer)
- **scikit-learn** — CountVectorizer, Pipeline, classifiers, metrics
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization

---

## Key Learnings

- **Naive Bayes excels on text:** Despite simplicity, Multinomial NB achieves competitive F1 on bag-of-words features.
- **Preprocessing matters:** Combining lemmatization, stemming, and stopword removal improves feature quality.
- **Pipeline design:** scikit-learn Pipelines prevent data leakage by applying vectorization only on training data.
- **Class imbalance awareness:** Stress class recall (0.83) is higher than precision (0.70), indicating the model catches most stress cases.

---

## Future Work

- TF-IDF vectorization instead of raw counts
- Word embeddings (Word2Vec, GloVe) or transformer models (BERT)
- Hyperparameter tuning via GridSearchCV
- Deploy as a web demo with Gradio or Streamlit

---

## License

Academic project — Penn State University.
