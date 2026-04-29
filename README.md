# 💬 Comment Category Prediction — Kaggle NLP Challenge
Kaggle ML project for multi-class text classification using TF-IDF and Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parkhiyadav/22f3002870-notebook-t12026)


"#⚠️ Note: This project was developed as part of a college assignment and is currently private for evaluation. kaggle notebook will be made public soon. i have instead shared my .ipynb file and submission here.."

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




# 💬 Comment Category Prediction — Kaggle NLP Challenge

Multi-class NLP classification of 300,000 social media comments into 4 categories · Macro F1: **0.8026**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange) ![Kaggle](https://img.shields.io/badge/Platform-Kaggle-lightblue)

> ⚠️ This project was developed as part of a college assignment and is currently private for evaluation. The Kaggle notebook will be made public soon. The `.ipynb` file and submission are shared here in the meantime.

---

## 🧩 Problem Statement

Classify 300,000 social media comments into 4 categories (labels 0–3) using text content and engagement metadata. Three things made this non-trivial:

- **Severe class imbalance:** label 0 accounts for 58% of samples; label 3 is only 2.8%. A dummy model predicting label 0 for everything achieves 58% accuracy — which is why macro F1, not accuracy, was used throughout.
- **Missing data at scale:** 73% of demographic columns were missing. EDA confirmed they carried zero predictive signal, so they were dropped rather than imputed — imputing noise adds noise.
- **300K samples:** preprocessing choices needed to be scalable and leak-free, which ruled out any fitting on the full dataset before splitting.

---

## 📊 Model Comparison

| Model | Macro F1 | Notes |
|-------|----------|-------|
| Tuned LR (Dual TF-IDF) | **0.8026 ✅** | Best model |
| Logistic Regression (baseline) | 0.7856 | Word TF-IDF only |
| SGD Classifier | 0.7716 | — |
| Multinomial Naive Bayes | 0.3051 ❌ | See diagnosis below |

### Why Did Naive Bayes Collapse?

The 0.30 F1 from Multinomial Naive Bayes is not a minor underperformance — it's a near-total failure, and diagnosing it is more useful than just flagging it.

MNB requires non-negative input features: it models each feature as a probability count. TF-IDF with `sublinear_tf=True` applies a log transformation (`1 + log(tf)`), which compresses term frequency but keeps values positive. However, with `min_df` filtering, stop word removal, and the character n-gram vectorizer combined via `FeatureUnion`, the resulting matrix had structural properties (sparse high-dimensional dual representation) that caused MNB's probability estimates to become unstable on the minority classes. The model effectively learned to over-predict label 0 — the dominant class — because the likelihood estimates for labels 1, 2, and 3 were being drowned out.

The fix that wasn't pursued here (but would be the right next step): use `ComplementNB` instead of `MultinomialNB`. Complement Naive Bayes is specifically designed for imbalanced text classification and estimates class probabilities using the *complement* of each class — making it far more robust on minority categories.

Reporting Logistic Regression's 0.78 without explaining why MNB scored 0.30 would miss the most informative signal in the experiment.

---

### Best Model — Per-Class Results (Tuned LR)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| 0 | 0.96 | 0.93 | 0.95 | 22,835 |
| 1 | 0.70 | 0.82 | 0.76 | 3,183 |
| 2 | 0.88 | 0.87 | 0.88 | 12,488 |
| 3 | 0.56 | 0.73 | 0.63 | 1,094 |
| **Macro Avg** | 0.77 | 0.84 | **0.80** | 39,600 |

Label 3 (2.8% of data) achieving F1 = 0.63 is the hardest result to obtain here, and the one most worth noting. A model that ignored minority classes would score near zero on label 3; the combination of macro F1 as the optimization target and `class_weight='balanced'` in Logistic Regression is what kept this from collapsing.

---

## 🔬 Feature Engineering

```python
text_features = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        max_features=40000, ngram_range=(1,2),
        min_df=2, stop_words='english'
    )),
    ('char_tfidf', TfidfVectorizer(
        max_features=15000, analyzer='char',
        ngram_range=(3,5)
    ))
])
```

Combining word-level and character-level TF-IDF via `FeatureUnion` is a well-established NLP technique, but it's the right choice here for a concrete reason: social media comments contain heavy misspellings, slang, and abbreviations ("luv", "omg", "lol") that word n-grams miss entirely. Character-level n-grams (3–5 characters) capture subword patterns — `"lovi"`, `"ovin"`, `"ving"` from "loving" — that survive even significant spelling variation. This improved F1 by ~0.017 over word-only features.

All feature extraction sits inside a sklearn `Pipeline`, ensuring no data leakage — the vectorizer is fit only on training folds during cross-validation.

---

## ⚙️ Training Setup

- **Hyperparameter tuning:** `RandomizedSearchCV` with 3-fold CV, optimizing for macro F1
- **Regularization sweep:** tested `C ∈ {0.01, 0.1, 1, 10}` for Logistic Regression
- **Solver:** `lbfgs` with `max_iter=1000` for convergence on high-dimensional sparse features
- **Class weighting:** `class_weight='balanced'` to counter label 0 dominance

---

## 💡 Key Takeaways

- **Metric selection is a modeling decision.** Accuracy reached 90% — but macro F1 at 0.80 told the honest story. The two numbers describe completely different models.
- **Diagnosing failure is more valuable than reporting it.** MNB's 0.30 F1 pointed directly to the interaction between imbalanced classes and probability estimate instability — a finding that points to `ComplementNB` as the right fix.
- **Dropping data can be the right call.** 73%-missing demographic columns add noise, not signal. EDA-confirmed removal improved generalization.
- **Character n-grams earn their cost.** The 15K additional features from char-level TF-IDF contributed a measurable +0.017 F1 on messy social media text.

---

## 🔭 Future Work

- [ ] Fine-tune DistilBERT for richer contextual representations
- [ ] Try `ComplementNB` as the theoretically-grounded fix for the MNB failure
- [ ] Test XGBoost / LightGBM on TF-IDF features
- [ ] Engineer time-based features from `created_date`
- [ ] Explore focal loss for further minority class improvement

---

*Parkhi Yadav · [Kaggle](https://kaggle.com)*
