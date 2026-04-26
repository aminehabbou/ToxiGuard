# ToxiGuard 🛡️

> **Toxic Comment Detection, Explanation, and Constructive Alternative Generation**  
> A multi-source NLP pipeline combining fine-tuned BERT classification with GPT-2 generative explanation.

---

## Table of Contents

- [Overview](#overview)
- [Results Summary](#results-summary)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Sample Output](#sample-output)
- [References](#references)

---

## Overview

ToxiGuard is an end-to-end NLP system designed to go beyond binary toxicity classification. Given an online comment, ToxiGuard:

1. **Classifies** whether the comment is toxic or non-toxic using a fine-tuned **BERT-base** encoder
2. **Explains** *why* a comment is toxic using a fine-tuned **GPT-2** decoder
3. **Suggests** three constructive alternative phrasings that preserve the author's intent while removing harmful content

The system was developed across four progressive modelling stages, from a TF-IDF Logistic Regression baseline through to a dual-headed transformer architecture, trained and evaluated on a curated multi-source corpus of **52,768** online comments.

---

## Results Summary

| Model | Test Set | Accuracy | F1-score | AUC |
|---|---|---|---|---|
| Logistic Regression (TF-IDF baseline) | 10,005 | 92.37% | 0.924 | — |
| KNN (engineered features) | 10,005 | 81.20% | 0.799 | 0.887 |
| Random Forest (engineered features) | 10,005 | 88.60% | 0.881 | 0.957 |
| SVM (engineered features) | 10,005 | 94.29% | 0.938 | 0.977 |
| **BERT fine-tuned** | 10,554 | **96.09%** | **0.961** | — |
| **ToxiGuard (BERT + GPT-2)** | 10,554 | **96.09%** | **0.961** | — |

> All metrics are macro-averaged. BERT and ToxiGuard use the full test partition; classical models use the 10,005-sample balanced test set.

---

## Architecture

```
Input Comment
      │
      ▼
┌─────────────────────────┐
│   BERT-base-uncased     │  ← Fine-tuned classification head
│   (110M parameters)     │
└─────────────────────────┘
      │
      ├── Predicted: NON-TOXIC ──► "Comment is safe."
      │
      └── Predicted: TOXIC (+ confidence score)
                │
                ▼
      ┌──────────────────┐
      │  GPT-2 (117M)    │  ← Fine-tuned generative head
      │  Explanation +   │
      │  3 Alternatives  │
      └──────────────────┘
```

**BERT Classification Head**
- Base model: `bert-base-uncased`
- Fine-tuned for 3 epochs, learning rate 2e-5, batch size 32
- Max sequence length: 128 tokens
- Output: binary label (0/1) + confidence score

**GPT-2 Generative Head**
- Base model: `gpt2` (117M)
- Fine-tuned for 2 epochs, learning rate 5e-5
- Trained on a structured template mapping toxic comments to category-specific explanations and rewrites
- Invoked only when BERT predicts toxic (confidence ≥ 0.5)

---

## Dataset

The corpus was assembled from three publicly available sources:

| Source | Samples | Toxic % | Domain |
|---|---|---|---|
| [Jigsaw Civil Comments](https://huggingface.co/datasets/google/civil_comments) | 20,000 | ~5% | News comments |
| [Davidson Hate Speech](https://github.com/t-davidson/hate-speech-and-offensive-language) | 24,783 | ~84% | Twitter |
| [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) | ~8,200 | 50% | Parallel detoxification |
| **Total (after cleaning)** | **52,768** | **47.4%** | Multi-domain |

**Preprocessing steps:**
- Removed duplicates, rows with missing `text` or `toxic` fields, and texts shorter than 10 characters
- Balanced to 50,022 samples (25,011 per class) via random undersampling
- Stratified 80/20 train-test split → **40,017 train / 10,005 test**

---

## Project Structure

```
toxiguard/
│
├── scripts/
│   ├── download_data.py              # Downloads all three source datasets
│   ├── combine_and_clean.py          # Merges, deduplicates, and cleans raw CSVs
│   ├── train_test_datasets.py        # Balances and splits into train/test
│   └── baseline_classifier.py       # TF-IDF + Logistic Regression baseline
│
├── notebooks/
│   ├── toxicity_eda_analysis.ipynb           # Exploratory data analysis
│   ├── features_correlation.ipynb            # Feature engineering & correlation matrices
│   ├── SVM_KNN_RandomForest_Optimization.ipynb  # Classical model grid search
│   └── BERT_with_explanation_alternative_comment_generator.ipynb  # ToxiGuard (BERT + GPT-2)
│
├── data/
│   ├── raw/                          # Downloaded source CSVs (gitignored)
│   └── processed/                    # Cleaned and split datasets (gitignored)
│
├── models/                           # Saved model weights (gitignored)
│   ├── toxic_baseline_classifier.pkl
│   └── vectorizer.pkl
│
├── results/                          # Evaluation outputs
│   ├── metrics_*.csv
│   ├── predictions_*.csv
│   ├── probabilities_*.csv
│   ├── misclassified_*.csv
│   ├── svm_*/
│   ├── knn_*/
│   └── rf_*/
│
└── README.md
```

---

## Pipeline Walkthrough

The project follows four stages in order:

### Stage 1 — Data Collection & Preprocessing
```bash
python scripts/download_data.py       # Downloads ~52K samples from 3 sources
python scripts/combine_and_clean.py   # Cleans and deduplicates
python scripts/train_test_datasets.py # Balances and splits
```

### Stage 2 — Exploratory Data Analysis
Open and run `notebooks/toxicity_eda_analysis.ipynb`.

Covers: class distribution, text length analysis, sentiment distributions, punctuation patterns, readability scores (Flesch, Fog, SMOG), word clouds, unigram/bigram/trigram frequency analysis, and source comparison.

### Stage 3 — Feature Engineering & Classical Models
Open and run `notebooks/features_correlation.ipynb` first (feature engineering and correlation analysis), then `notebooks/SVM_KNN_RandomForest_Optimization.ipynb` (grid search and evaluation).

**Engineered features (15 total):**
- Text length: `char_count`, `word_count`, `sentence_count`
- Punctuation: `exclamation_count`, `question_count`, `period_count`
- Capitalisation: `all_caps_words`, `caps_ratio`
- Sentiment (VADER): `sentiment_compound`, `sentiment_positive`, `sentiment_negative`, `sentiment_neutral`
- Readability: `flesch_score`, `fog_index`, `smog_index`

**Key finding:** `sentiment_negative` is the single strongest predictor of toxicity (Pearson r = 0.46). Text length features are the next most informative group.

### Stage 4 — ToxiGuard (BERT + GPT-2)
Open and run `notebooks/BERT_with_explanation_alternative_comment_generator.ipynb`.

This notebook trains the BERT classification head, fine-tunes the GPT-2 generative head on the structured template, assembles the inference pipeline, and evaluates on the held-out test set.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/toxiguard.git
cd toxiguard

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0
transformers>=4.30
datasets>=2.12
scikit-learn>=1.2
pandas>=2.0
numpy>=1.24
nltk>=3.8
textstat>=0.7
vaderSentiment>=3.3
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9
jupyter
```

> **Note:** Training the BERT and GPT-2 components requires a GPU. CPU-only inference is possible but significantly slower.

---

## Usage

### Run the baseline classifier
```bash
python scripts/baseline_classifier.py
```
Outputs timestamped CSVs to `results/`: metrics, predictions, probabilities, misclassified examples.

### Run ToxiGuard inference (from notebook)
After training, the inference pipeline can be called as:

```python
from pipeline import ToxiGuard  # see notebook for implementation

model = ToxiGuard(
    bert_path="models/bert_classifier/",
    gpt2_path="models/gpt2_explainer/"
)

result = model.predict("You're such an idiot, nobody wants to hear from you.")
print(result)
```

**Example output:**
```
{
  "label": "TOXIC",
  "confidence": 0.994,
  "explanation": "the language here is unnecessarily harsh. Strong words can make 
                  people defensive and stop listening to what you have to say.",
  "alternatives": [
    "I feel strongly about this, and here's my honest view.",
    "I'm really frustrated by this situation.",
    "I strongly disagree with that position."
  ]
}
```

For non-toxic comments, the pipeline returns:
```
{
  "label": "NON-TOXIC",
  "confidence": 0.981,
  "message": "Comment is safe."
}
```

---

## Experiments

### Hyperparameter Search Results

| Model | Best Parameters | CV F1 |
|---|---|---|
| SVM | kernel=linear, C=3.234, γ=0.098 | 0.9348 |
| Random Forest | n_estimators=169, max_depth=30, min_split=7, max_features=sqrt | 0.8820 |
| KNN | k=11, weights=uniform, metric=euclidean, algo=ball_tree | 0.7947 |

### BERT — Per-class Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-Toxic | 0.9616 | 0.9642 | 0.9629 | 5,552 |
| Toxic | 0.9601 | 0.9572 | 0.9587 | 5,002 |
| **Macro Avg** | **0.9608** | **0.9607** | **0.9608** | **10,554** |

### Error Analysis

Of the 413 misclassified samples (3.91% error rate):
- **214 False Negatives** (toxic missed) — mean confidence 0.778
- **199 False Positives** (non-toxic flagged) — mean confidence 0.730

The low confidence scores on misclassified samples suggest a confidence-threshold review layer (e.g., flag predictions < 0.70 for human review) could recover a substantial fraction of errors.

---

## Sample Output

### Correctly Classified — Toxic with GPT-2 Explanation

```
Input:    "RT @DeePaPi1800: That outside dick keep dem hoes sick..."

BERT:     TOXIC | Confidence: 99.6%

GPT-2:    it includes threatening language. Wishing harm on others
          creates a hostile environment and is never acceptable.
          Consider saying any of the following:
            - I'm angry, but I want to address this constructively.
            - This really upset me, and I need to say that clearly.
            - Let's deal with the issue directly instead of escalating it.
```

### Failure Case — False Negative (diluted toxicity)

```
Input:    [~300-word civic complaint concluding with "Motherf***ers!"]

BERT:     NON-TOXIC | Confidence: 97.4%   ← WRONG

Analysis: A single expletive embedded in a long non-toxic argument
          is down-weighted by BERT's contextual aggregation.
```

---

## References

Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. In Companion Proceedings of The 2019 World Wide Web Conference (WWW '19 Companion), pp. 491–500. ACM. https://doi.org/10.1145/3308560.3317593
Dale, D., Voronov, A., Dementieva, D., Logacheva, V., Kozlova, O., Semenov, N., & Panchenko, A. (2021). Text Detoxification using Large Pre-trained Neural Models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), pp. 7979–7996. Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.629
Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. In Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM 2017), pp. 512–515. AAAI Press. https://doi.org/10.1609/icwsm.v11i1.14955
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019), Volume 1 (Long and Short Papers), pp. 4171–4186. Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423
Hanu, L., & Unitary Team. (2020). Detoxify. GitHub repository. Available at: https://github.com/unitaryai/detoxify
Logacheva, V., Dementieva, D., Ustyantsev, S., Moskovskiy, D., Dale, D., Krotova, I., Semenov, N., & Panchenko, A. (2022). ParaDetox: Detoxification with Parallel Data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022), Volume 1 (Long Papers), pp. 6804–6818. Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.acl-long.469
Nobata, C., Tetreault, J., Thomas, A., Mehdad, Y., & Chang, Y. (2016). Abusive Language Detection in Online User Content. In Proceedings of the 25th International Conference on World Wide Web (WWW 2016), pp. 145–153. ACM. https://doi.org/10.1145/2872427.2883062

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, pp. 2825–2830.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog, 1(8). Available at: https://openai.com/research/language-unsupervised

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS 2017), Volume 30. Curran Associates.

---

## Ethical Notice

ToxiGuard is a research prototype. Known limitations include:

- **Demographic bias:** The model may exhibit higher false positive rates on comments containing identity terms associated with marginalised groups, reflecting biases in the source annotations.
- **Domain sensitivity:** Performance on non-English text, gaming platforms, or messaging applications is unknown.
- **Binary labels:** The system does not distinguish between severity levels of toxic content.

This system should **not** be deployed as a standalone moderation tool without a human review layer and a formal bias audit.

---

*Built as part of a research project on NLP-based content moderation.*
