# SafeSignal

**Detecting hidden drug side-effects from patient reviews using NLP.**
Course project for *Natural Language Processing (0304220301)*, Symbiosis Artificial Intelligence Institute.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/SafeSignal/blob/main/notebooks/SafeSignal_Colab.ipynb)

Patients describe medicine side-effects on health forums in their own words. SafeSignal reads those posts, finds the drugs and the reactions linked to them, and turns free text into a simple early-warning signal.

![Pipeline](assets/pipeline.png)

## What the app can do

| Tab | What you can try | Syllabus unit |
|---|---|---|
| Post Analyzer | Highlight Drug / ADR / Disease / Symptom / Finding, safety-signal score | 1, 2 |
| Batch Signal Detection | Upload a CSV, get drug-reaction pairs, charts, downloadable results | 2 |
| Preprocessing Lab | Tokenise, stop-words, stemming, lemmatising, regex tester | 1, 2 |
| Spelling & Edit Distance | Fix misspelt drug names, see the Levenshtein matrix | 2 |
| N-gram Lab | Unigram/bigram/trigram model, add-k smoothing, perplexity, text generation | 2 |
| Features & Embeddings | Bag-of-Words, TF-IDF, One-Hot, LSA embeddings, similarity search | 3 |
| Sentiment & Opinion | Lexicon sentiment, opinion-word highlighting | 4 |
| Syntax & Semantics | POS tags, dependency parse, WordNet, word-sense disambiguation | 4 |
| LLM Assistant | FLAN-T5 extracts side-effects, summarises, rates severity | 5 |
| Model Report | CADEC results, formal-vs-informal audit, ethics notes | 5 |

## Quick start

**Google Colab (easiest):** open `notebooks/SafeSignal_Colab.ipynb`, choose *Runtime > Run all*, then open the `gradio.live` link.

**Locally:**
```bash
git clone https://github.com/YOUR_USERNAME/SafeSignal.git
cd SafeSignal
pip install -r requirements.txt
python app.py
```
Run the tests with `pytest tests/`.

## Results (CADEC test set, 250 posts)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| Drug | 0.94 | 0.88 | 0.91 |
| ADR | 0.65 | 0.65 | 0.65 |
| Finding | 0.43 | 0.29 | 0.35 |
| Disease | 0.33 | 0.12 | 0.18 |
| Symptom | 0.50 | 0.08 | 0.14 |

![F1](assets/f1_by_entity.png)
![Audit](assets/formal_vs_informal.png)

The audit shows that Disease detection drops from 0.26 on formal text to 0.00 on informal text.

> **Note on models.** The numbers above come from the spaCy NER trained on CADEC. The app uses that model automatically if you place it in `models/safesignal_cadec/`. Without it, the app trains a small *demo* model on synthetic data so everything still runs, and a rule-based engine with fuzzy drug matching covers the rest.

## Repository layout

```
SafeSignal/
├── app.py                  # full application (all features)
├── notebooks/              # Colab notebook (generated from app.py)
├── data/                   # synthetic sample posts (CADEC is not redistributed)
├── models/                 # drop the trained spaCy model here
├── tests/                  # pytest checks for the core functions
├── docs/                   # project report and pitch deck
├── assets/                 # figures used in this README
├── requirements.txt
└── setup_github.sh
```

## Ethics and limitations
Not medical advice. The tool supports human reviewers and must not replace them. Forum posts are personal health information. The training data covers one forum, 12 drugs and English only, and the model is weaker on informal writing.

## Author
Sanobar Latif Shaikh (PRN 25030421039) - Symbiosis Artificial Intelligence Institute

## Licence
MIT. CADEC data: Karimi et al. (2015), CSIRO.
