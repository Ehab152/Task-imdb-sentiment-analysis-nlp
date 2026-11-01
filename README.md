# IMDb Sentiment Analysis (NLP Mini Project)

> 📦 **Dataset:** [IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
> — original source: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
> The CSV (~66 MB) is **not included** in this repo. Download it and set `DATA_PATH` in the first cell of the notebook.

Binary sentiment classification (positive / negative) on 50,000 IMDb movie reviews. The project compares a classical **TF-IDF + Logistic Regression** baseline against several neural sequence models built with Keras: **RNN, LSTM, Bidirectional LSTM, and GRU**.

## Dataset

- 50,000 reviews, two columns: `review`, `sentiment`
- Perfectly balanced: 25,000 positive / 25,000 negative, no missing values
- Stratified 80/20 split → 40,000 training / 10,000 test reviews (`random_state=42`)

## Pipeline

1. **Cleaning:** lowercase, strip HTML tags (`<br />` etc.), remove non-alphanumeric characters (apostrophes kept), normalize whitespace
2. **TF-IDF baseline:** `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)` + Logistic Regression
3. **Word2Vec:** 100-dimensional embeddings trained with Gensim (exploration of word similarity)
4. **Neural models:** Tokenizer (20,000-word vocabulary) → padded sequences (length 250) → Embedding(128) → RNN / LSTM / BiLSTM / GRU → Dense layers → sigmoid. Trained with Adam, early stopping, and a 15% validation split
5. **Extras:** dropout comparison, vocabulary/sequence-length experiment, LSTM error analysis

## Results (10,000-review test set)

| Model | Representation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | TF-IDF | **0.9077** | 0.9033 | 0.9132 | **0.9082** |
| RNN | Embedding | 0.5004 | 0.5227 | 0.0092 | 0.0181 |
| LSTM | Embedding | 0.8632 | 0.8847 | 0.8352 | 0.8593 |
| Bidirectional LSTM | Embedding | 0.8746 | 0.9031 | 0.8392 | 0.8700 |
| GRU | Embedding | 0.8768 | 0.8854 | 0.8656 | 0.8754 |

### TF-IDF experiment (features and n-grams)

| max_features | ngram_range | Accuracy | F1 |
|---|---|---|---|
| 10,000 | (1, 1) | 0.9007 | 0.9012 |
| 20,000 | (1, 1) | 0.9016 | 0.9022 |
| 20,000 | (1, 2) | 0.9077 | 0.9082 |
| 30,000 | (1, 2) | 0.9079 | 0.9086 |

### Vocabulary size and sequence length experiment

Small GRU (embedding 64, 32 units), trained for only 2 epochs per setting, so the numbers are indicative only.

| Vocabulary size | Max sequence length | Accuracy | F1 |
|---|---|---|---|
| 10,000 | 150 | 0.8187 | 0.8360 |
| 20,000 | 200 | 0.7725 | 0.7861 |
| 20,000 | 250 | 0.5301 | 0.6472 |
| 30,000 | 300 | 0.5462 | 0.2999 |

## Notes

- The simple RNN stayed close to chance level (≈50% accuracy), while the gated models (LSTM, BiLSTM, GRU) learned well.
- The pretrained Transformer section (`distilbert-base-uncased-finetuned-sst-2-english`) needs `transformers` and `torch`. It was not executed in the saved notebook, so no Transformer results are reported above.

## Project structure

```
├── IMDB_NLP_Mini_Project.ipynb     # full analysis and experiments
├── results/
│   ├── IMDB_NLP_Final_Results.csv
│   ├── TFIDF_Experiments.csv
│   └── Sequence_Experiments.csv
└── README.md
```

## How to run

1. Download `IMDB Dataset.csv` from the Kaggle link at the top.
2. Open the notebook in Google Colab or Jupyter.
3. Set `DATA_PATH` in the first code cell to where you saved the CSV (on Colab: upload it to `/content/`).
4. Run all cells.

## Requirements

```
pandas numpy matplotlib scikit-learn tensorflow gensim
# optional, for the Transformer section:
transformers torch
```
