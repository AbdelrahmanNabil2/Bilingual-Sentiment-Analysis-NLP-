# NLP Final Project — Arabic & English Text Classification

End-to-end NLP project comparing **classical machine learning** and **transformer (BERT)** models on two classification tasks:

1. **Arabic tweet sentiment analysis** (Negative / Neutral / Positive)
2. **English mental-health status classification** (7 classes)

Main notebook: [`notebooks/Final_Project_NLP.ipynb`](notebooks/Final_Project_NLP.ipynb)

---

## Table of contents

- [Project overview](#project-overview)
- [Datasets](#datasets)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Training setup](#training-setup)
- [Results](#results)
- [Project structure](#project-structure)
- [Setup & run](#setup--run)
- [Notes & limitations](#notes--limitations)

---

## Project overview

| Stage | Task | Approach |
|-------|------|----------|
| A | Arabic sentiment | TF-IDF / BoW / Word2Vec + LogReg, SVM, MultinomialNB |
| B | Arabic sentiment | Fine-tuned AraBERT (`bert-base-arabertv02-twitter`) |
| C | English status | Fine-tuned BERT (`bert-base-uncased`) |

**Key finding:** On Arabic sentiment, AraBERT (**F1 ≈ 0.84**) clearly beats the best classical pipeline (**TF-IDF + Logistic Regression, F1 ≈ 0.77**).

---

## Datasets

Datasets are **not committed** to this repository (see [`.gitignore`](.gitignore)). Place them in [`data/`](data/) as described in [`data/README.md`](data/README.md).

### 1. Arabic tweets — `arabic_tweets.csv`

| Property | Detail |
|----------|--------|
| **Task** | 3-class sentiment classification |
| **Language** | Arabic |
| **Approx. size** | ~18,678 labeled tweets (after dropping missing text/labels) |
| **Text column** | `Tweet_text` |
| **Label column** | `Sentiment_label` |
| **Labels** | `Negative`, `Neutral`, `Positive` |

**Additional columns** (present in the source file; optional for training):

| Column | Description |
|--------|-------------|
| `#Tweet_ID` / tweet id | Unique tweet identifier |
| `Topic` | Topic category (e.g. Event, Entity, Long_Standing) |
| `Sentiment_label_confidence` | Annotator / model confidence for sentiment |
| `Speech_act_label` | Speech act (e.g. Assertion, Expression) |
| `Speech_act_label_confidence` | Confidence for speech-act label |

**Splits used in this project**

| Experiment | Split | Stratified |
|------------|-------|------------|
| Traditional ML | 80% train / 20% test | No (random split, `random_state=42`) |
| AraBERT | 90% train / 10% validation | Yes (`stratify=label`) |

> If your file is named `arabic tweets.csv` (with a space), rename it to `arabic_tweets.csv`.

### 2. English mental-health statements — `data.csv`

| Property | Detail |
|----------|--------|
| **Task** | Multi-class status / condition classification |
| **Language** | English |
| **Approx. size** | ~52,690 rows (validation support ≈ 5,269 at 10%) |
| **Text column** | `statement` |
| **Label column** | `status` |

**Label set (7 classes)**

| Label | Approx. validation support (from project run) |
|-------|-----------------------------------------------|
| Anxiety | 384 |
| Bipolar | 278 |
| Depression | 1,541 |
| Normal | 1,634 |
| Personality disorder | 108 |
| Stress | 259 |
| Suicidal | 1,065 |

**Split:** 90% train / 10% validation, stratified by label (`random_state=42`).

---

## Models

### A. Traditional ML (Arabic)

**Feature representations**

| Representation | Library / method | Notes |
|----------------|------------------|-------|
| **TF-IDF** | `TfidfVectorizer` + custom Arabic tokenizer | Sparse bag-of-weighted-tokens |
| **Bag of Words (BoW)** | `CountVectorizer` + same tokenizer | Raw term counts |
| **Word2Vec** | `gensim.Word2Vec` | `vector_size=100`, `window=5`, `min_count=1`; document = mean of word vectors |

**Classifiers**

| Model | Config | Used with |
|-------|--------|-----------|
| Logistic Regression | `max_iter=1000` | TF-IDF, BoW, Word2Vec |
| SVM (`SVC`) | sklearn defaults | TF-IDF, BoW, Word2Vec |
| Multinomial Naive Bayes | sklearn defaults | TF-IDF, BoW only (not Word2Vec) |

**Metrics:** Accuracy and weighted F1 on the held-out test set.

### B. AraBERT — Arabic sentiment

| Property | Value |
|----------|-------|
| **Base model** | [`aubmindlab/bert-base-arabertv02-twitter`](https://huggingface.co/aubmindlab/bert-base-arabertv02-twitter) |
| **Architecture** | BERT (Arabic, Twitter-domain pretrained) |
| **Head** | Sequence classification, **3 labels** |
| **Label mapping** | Negative→0, Neutral→1, Positive→2 |
| **Tokenizer** | Matching AutoTokenizer from the same checkpoint |
| **Max sequence length** | 128 |
| **Padding** | `max_length` |
| **Truncation** | Enabled |

Why this model: pretrained on Arabic social-media text, which matches the tweet domain better than a general Arabic BERT.

### C. BERT-base — English status

| Property | Value |
|----------|-------|
| **Base model** | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) |
| **Architecture** | BERT-base (uncased English) |
| **Head** | Sequence classification, **7 labels** (dynamic from `status`) |
| **Tokenizer** | `bert-base-uncased` WordPiece |
| **Max sequence length** | 128 |
| **Padding** | `max_length` |
| **Truncation** | Enabled |
| **Input text** | Lemmatized / stopword-filtered `cleaned_statement` |

---

## Preprocessing

### Arabic (traditional ML)

1. Character normalization (`إأآا`→`ا`, `ى`→`ي`, `ة`→`ه`, etc.)
2. Remove punctuation, digits, tatweel (`ـ`)
3. Tokenize with `regexp_tokenize`
4. Remove Arabic stopwords (`nltk` Arabic stopword list)

### Arabic (AraBERT)

1. Remove URLs, `@mentions`, `#` characters, emojis, tatweel
2. Strip Arabic punctuation
3. Same style of Arabic character normalization
4. Collapse whitespace  
*(Stopword removal is intentionally lighter here so the transformer can use full context.)*

### English (BERT)

1. Lowercase
2. Remove punctuation
3. Tokenize (`word_tokenize`)
4. Remove English stopwords
5. Lemmatize (`WordNetLemmatizer`)
6. Join tokens back into a cleaned string for BERT

---

## Training setup

Shared Hugging Face `Trainer` settings for both transformer experiments:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | `2e-5` |
| Train batch size | 16 |
| Eval batch size | 32 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Eval / save strategy | Every epoch |
| Best model selection | Highest **weighted F1** |
| Seed | 42 |
| Checkpoint limit | 2 |
| Logging | Local only (`report_to="none"`) |

**Saved outputs (gitignored)**

| Run | Output directory |
|-----|------------------|
| AraBERT | `./arabic_sentiment_finetuned_final` |
| English BERT | `./english_emotion_model_final` |

**Hardware:** GPU strongly recommended for fine-tuning; CPU works but is slow.

---

## Results

Numbers below are from the completed project runs (stored as reference targets in the notebook). Re-runs may vary slightly with hardware / library versions.

### Traditional ML — Arabic sentiment (full table)

| Rank | Representation | Model | Accuracy | Weighted F1 |
|------|----------------|-------|----------|-------------|
| 1 | TF-IDF | LogisticRegression | **0.771** | **0.766** |
| 2 | BoW | LogisticRegression | 0.767 | 0.765 |
| 3 | TF-IDF | SVC | 0.766 | 0.761 |
| 4 | BoW | SVC | 0.759 | 0.754 |
| 5 | BoW | MultinomialNB | 0.759 | 0.753 |
| 6 | TF-IDF | MultinomialNB | 0.738 | 0.716 |
| 7 | Word2Vec | SVC | 0.681 | 0.667 |
| 8 | Word2Vec | LogisticRegression | 0.666 | 0.651 |

Word2Vec mean-pooling underperformed on short tweets (sparse context, `min_count=1`).

### AraBERT — Arabic sentiment

| Metric | Score |
|--------|-------|
| Accuracy | **0.843** |
| Weighted F1 | **0.843** |
| Weighted precision | 0.843 |
| Weighted recall | 0.843 |
| Eval loss | 0.622 |

**Per-class (validation)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Negative | 0.90 | 0.88 | 0.89 | 739 |
| Neutral | 0.84 | 0.85 | 0.85 | 689 |
| Positive | 0.75 | 0.76 | 0.76 | 440 |
| **Weighted avg** | **0.84** | **0.84** | **0.84** | **1,868** |

### English BERT — mental-health status

| Metric | Score |
|--------|-------|
| Accuracy | **0.814** |
| Weighted F1 | **0.814** |
| Weighted precision | 0.817 |
| Weighted recall | 0.814 |
| Eval loss | 0.506 |

**Per-class (validation)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Anxiety | 0.87 | 0.87 | 0.87 | 384 |
| Bipolar | 0.84 | 0.83 | 0.84 | 278 |
| Depression | 0.81 | 0.70 | 0.75 | 1,541 |
| Normal | 0.93 | 0.96 | 0.94 | 1,634 |
| Personality disorder | 0.73 | 0.74 | 0.73 | 108 |
| Stress | 0.68 | 0.68 | 0.68 | 259 |
| Suicidal | 0.67 | 0.78 | 0.72 | 1,065 |
| **Weighted avg** | **0.82** | **0.81** | **0.81** | **5,269** |

### Head-to-head summary

| Approach | Task | Best Accuracy | Best F1 |
|----------|------|---------------|---------|
| TF-IDF + Logistic Regression | Arabic sentiment | 0.77 | 0.77 |
| **AraBERT** | Arabic sentiment | **0.84** | **0.84** |
| **BERT-base** | English status | **0.81** | **0.81** |

---

## Project structure

```text
NLP/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md              # dataset file format & placement
│   ├── arabic_tweets.csv      # you add locally (ignored by git)
│   └── data.csv               # you add locally (ignored by git)
└── notebooks/
    └── Final_Project_NLP.ipynb
```

---

## Setup & run

### 1. Clone & environment

```bash
git clone <your-repo-url>
cd NLP

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA GPU build of PyTorch, install from [pytorch.org](https://pytorch.org) instead of (or after) the default wheel.

### 2. Add data

```bash
# expected paths
data/arabic_tweets.csv
data/data.csv
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/Final_Project_NLP.ipynb
# or: jupyter lab
```

**Google Colab**

1. Upload the notebook and both CSVs (or mount Drive).
2. If files land in `/content`, the notebook falls back to that path automatically.
3. Use a **GPU** runtime for AraBERT / BERT fine-tuning.

---

## Notes & limitations

- Classical and transformer Arabic experiments use **different split ratios** (80/20 vs 90/10). Compare trends, not absolute test scores one-to-one.
- Class imbalance exists (especially English: Depression / Normal / Suicidal dominate; Personality disorder is rare).
- Word2Vec here is trained only on the training fold (not pretrained embeddings).
- Fine-tuning is sensitive to GPU, batch size, and library versions; expect small metric variance on re-runs.
- Model checkpoints and logs are gitignored — retrain or download artifacts separately if you need them.

## License / academic use

This repository is intended for coursework and research demonstration. Respect the license and terms of any dataset you add locally, and of the Hugging Face model checkpoints you download.
