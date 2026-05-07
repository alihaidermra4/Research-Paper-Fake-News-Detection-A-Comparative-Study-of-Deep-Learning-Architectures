# Assessing the Effectiveness of Modern Deep Learning Architectures in Multi-Label News Categorization
## Overview

This repository contains the implementation for a systematic benchmarking study of deep learning sequence models applied to binary fake news classification. Seven architectures — CNN, CNN+RNN, Bidirectional RNN, LSTM, Bidirectional LSTM, GRU, and Bidirectional GRU — are evaluated on the Kaggle Fake-and-Real News Dataset to identify the most effective architecture for this task.

**Published in:** International Journal of Innovative Science and Technology (IJIST)
**DOI / Article:** https://journal.50sea.com/index.php/IJIST/article/view/1857

---

## Dataset

- **Source:** [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) (Clément Bisaillon)
- **Files:** `Fake.csv` (label = 0) and `True.csv` (label = 1)
- **Combined size:** ~44,000 news articles
- **Task:** Binary classification (fake vs. real)
- **Split:** 80% train / 20% test

---

## Preprocessing Pipeline

1. Lowercase conversion
2. Removal of URLs, HTML tags, and non-alphabetic characters
3. NLTK stopword removal
4. Keras `Tokenizer` (vocabulary size = 5,000)
5. Sequence padding to `max_len = 500`

---

## Models

All models share a common Keras `Embedding` layer (input_dim=5000, output_dim=128) trained end-to-end. Early stopping (patience=3, monitor=`val_loss`) is applied across all experiments (max 50 epochs, batch size=64, 20% validation split).

| # | Architecture | Test Accuracy |
|---|---|---|
| 1 | CNN (Conv1D → MaxPool → Flatten → Dense) | 99.82% |
| 2 | CNN + SimpleRNN | 99.21% |
| 3 | Bidirectional SimpleRNN | 96.80% |
| 4 | LSTM | 99.51% |
| 5 | Bidirectional LSTM | 99.57% |
| 6 | GRU | 99.77% |
| 7 | **Bidirectional GRU** | **99.91%** ✓ |

**Best model: Bidirectional GRU** — 99.91% accuracy.

---

## Repository Structure

```
.
├── fake-real-news-classification-dataset.ipynb   # Main experiment notebook
├── README.md
```

---

## Requirements

```
tensorflow>=2.x
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
gensim
```

Install via:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn nltk gensim
```

Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

---

## Usage

1. Download `Fake.csv` and `True.csv` from the Kaggle dataset link above.
2. Update the file paths in the notebook to match your local/Kaggle environment.
3. Run all cells sequentially. Each section trains and evaluates one architecture independently.

---

## Results Summary

The Bidirectional GRU outperforms all other architectures, confirming that bidirectional context capture and gated memory are jointly beneficial for news text classification. The plain Bidirectional RNN underperforms significantly (96.80%), suggesting that gating mechanisms (LSTM/GRU) are critical for this task.

---

## Citation

If you use this work, please cite the corresponding IJIST article:

```
[Author(s)]. Assessing the Effectiveness of Modern Deep Learning Architectures in Multi-Label News Categorization.
International Journal of Innovative Science and Technology (IJIST).
Available: https://journal.50sea.com/index.php/IJIST/article/view/1857
```
