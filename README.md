# 💬 Comment Category Prediction — Kaggle NLP Challenge
Kaggle ML project for multi-class text classification using TF-IDF and Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parkhiyadav/22f3002870-notebook-t12026)


"#⚠️ Note: This project was developed as part of a college assignment and is currently private for evaluation. kaggle notebook will be made public soon."

> Multi-class NLP classification of **300,000 social media comments** into 4 categories — Kaggle competition | **Macro F1: 0.8026**

---

## 🧩 Problem Statement

Classify comments into 4 labels (0–3) using text content and engagement metadata. Key challenges: severe class imbalance (label 0 = 58%, label 3 = 2.8%), 73% missing demographic data, and 300K samples requiring scalable preprocessing.

---

## 🚀 Highlights

- **Dual TF-IDF** — word-level (40K features) + character-level (15K features) via `FeatureUnion`
- **Data leakage prevention** — all preprocessing inside `sklearn` Pipelines
- **EDA-driven decisions** — dropped 73%-missing demographic columns after confirming zero predictive power
- **Hyperparameter tuning** — `RandomizedSearchCV` with 3-fold CV optimized for macro F1

---

## 📊 Model Comparison

| Model | Macro F1 |
|-------|----------|
| **Tuned Best Model** (Dual TF-IDF + LR) | **0.8026** ✅ |
| Logistic Regression (baseline) | 0.7856 |
| SGD Classifier | 0.7716 |
| Multinomial Naive Bayes | 0.3051 ❌ |

**Best Model — Validation Results:**

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.96 | 0.93 | 0.95 | 22,835 |
| 1 | 0.70 | 0.82 | 0.76 | 3,183 |
| 2 | 0.88 | 0.87 | 0.88 | 12,488 |
| 3 | 0.56 | 0.73 | 0.63 | 1,094 |
| **Macro Avg** | **0.77** | **0.84** | **0.80** | 39,600 |

> Macro F1 was chosen over accuracy because a dummy model always predicting label 0 would hit 58% accuracy — but be completely useless. Macro F1 treats all 4 classes equally.

---

## 🔬 Core Innovation — Dual TF-IDF

```python
text_features = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2, stop_words='english')),
    ('char_tfidf', TfidfVectorizer(max_features=15000, analyzer='char', ngram_range=(3,5)))
])
```

Character-level n-grams handle misspellings and slang common in online comments — a significant improvement over word-only models.


## 💡 Key Learnings

- **More features ≠ better model** — removing noisy columns improved F1 by ~0.017
- **Character n-grams** handle real-world messy text far better than word-only approaches
- **EDA should drive every decision** — no arbitrary choices in this pipeline
- **Metric selection matters** — accuracy was 90%, but macro F1 told the honest story

---

## 🔭 Future Work

- [ ] Fine-tune **DistilBERT** for richer text representations
- [ ] Try **XGBoost / LightGBM** on TF-IDF features
- [ ] Engineer time features from `created_date`
- [ ] Improve handling of class imbalance

---

## 👤 Author

**Parkhi Yadav**  [Kaggle](https://www.kaggle.com/code/parkhiyadav/22f3002870-notebook-t12026) ·

