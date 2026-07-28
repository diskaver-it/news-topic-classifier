# News Topic Classifier — and *why* it picked that topic

Classify a piece of text into one of 20 newsgroup topics, and — the part most
write-ups skip — show **which words drove the decision**. Built on the classic
[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) corpus (~18,800 Usenet
posts) with a deliberately interpretable TF-IDF + logistic-regression pipeline,
wrapped in a Streamlit demo.

```
┌── paste text ─────────────────────────────────────────────┐
│  "The spacecraft entered orbit around Mars after a         │
│   seven-month journey; NASA confirmed the burn..."         │
└───────────────────────────────────────────────────────────┘
        │
        ▼   TF-IDF (1–2 grams)  →  logistic regression
        │
   ┌────────────────────┐   ┌──────────────────────────────────┐
   │ sci.space   99%     │   │ why?  orbit  +2.20               │
   │ talk.mideast 0.4%   │   │       spacecraft +1.88           │
   │ sci.med      0.2%   │   │       nasa +1.67  mars +1.32     │
   └────────────────────┘   └──────────────────────────────────┘
```

---

## Results

Trained on message **content only**, evaluated on the official by-date test
split (7,532 posts the model never saw, written *later* than the training set):

| Metric | Score |
|---|---|
| Accuracy | **0.695** |
| Macro-F1 | **0.685** |
| Weighted-F1 | 0.695 |
| Classes | 20 |
| Regularisation `C` | 10.0 (chosen by CV) |

That is not a flashy number, and it is not supposed to be — see the leakage
section below for why the honest version of this task tops out far lower than
the 0.90+ you will see in most notebooks.

Per-topic F1 ranges from **0.88** (rec.sport.hockey — a topic with its own
unmistakable vocabulary) down to **0.38** (talk.religion.misc — a catch-all that
overlaps heavily with atheism and Christianity):

![Per-topic F1](reports/figures/per_class_f1.png)

---

## What makes this more than a fit-predict script

### 1. Every prediction is explained

Because the model is linear over TF-IDF features, the score for a topic is a sum
of per-word contributions:

```
score(topic) = bias + Σ  tfidf(word) × weight[topic, word]
                    word in text
```

So for any input we can rank the words by how hard they pushed the decision —
computed on *your* text, not read off a global table. Verified on real inputs:

| Input (paraphrased) | Predicted | Top contributing words |
|---|---|---|
| "spacecraft entered orbit around Mars, NASA confirmed the burn" | `sci.space` (99%) | orbit, spacecraft, nasa, mars |
| "the clipper chip is a government backdoor into encrypted messages" | `sci.crypt` (100%) | encryption, clipper, government, encrypted |
| "the goalie stopped 45 shots, went to overtime in the playoffs" | `rec.sport.hockey` (81%) | playoffs, shots, overtime, goalie |

This is the whole reason for choosing a linear model over something heavier: a
gradient-boosted or neural model would likely score a little higher and could
not hand you this explanation as cleanly. For a topic classifier, being able to
say *why* is worth more than a point or two of accuracy.

The same mechanism gives each topic's learned "definition" — its highest-weight
words — which doubles as a sanity check that the model learned subject matter and
not noise:

| Topic | Top words |
|---|---|
| sci.space | space, orbit, launch, nasa, spacecraft, moon |
| rec.sport.hockey | hockey, team, game, nhl, season |
| sci.crypt | encryption, clipper, key, nsa, government, security |

### 2. The metadata leak, measured

The raw posts carry their newsgroup name in the **headers**, recurring poster
signatures in the **footers**, and quoted parent text that is usually on-topic.
Train on all of that and accuracy jumps — but the model is reading metadata, not
understanding language. Most public 20 Newsgroups results quietly leave these in.

The shipped model strips them (`remove=('headers','footers','quotes')`), and the
cost of that honesty is measured rather than hand-waved
([`reports/leakage_comparison.json`](reports/leakage_comparison.json)):

| Same pipeline, trained on… | Accuracy | Macro-F1 |
|---|---|---|
| Content only | 0.694 | 0.679 |
| Content + headers/footers/quotes | 0.838 | 0.832 |
| **Inflation from the leak** | **+0.144** | **+0.153** |

Fourteen points of "accuracy" for free — and worthless, because a real post's
headers are exactly what you would not have if you were trying to route it in the
first place.

<sub>Both rows use the default `C=1.0` so the only thing that changes is the
feature set — this isolates the leak, not the tuning. The shipped model adds the
CV-tuned `C=10`, which lifts content-only accuracy to the 0.695 in the results
table above.</sub>

### 3. The mistakes are the *reasonable* kind

A 20×20 confusion matrix is hard to read, so the model's errors are summarised as
its most frequent (true → predicted) confusions. **All ten of the top confusions
are between sibling topics** — two `comp.*` groups, or atheism vs Christianity —
rather than wild misfires:

| True topic | Predicted as | Count | Siblings? |
|---|---|---|---|
| talk.politics.misc | talk.politics.guns | 88 | ✓ |
| talk.religion.misc | soc.religion.christian | 58 | ✓ |
| alt.atheism | soc.religion.christian | 49 | ✓ |
| comp.windows.x | comp.graphics | 48 | ✓ |
| rec.motorcycles | rec.autos | 41 | ✓ |

The block structure is visible in the confusion matrix — bright clusters line up
with the super-categories (`comp.*`, the `talk.*`/religion group):

![Confusion matrix](reports/figures/confusion_matrix.png)

A classifier that confuses hockey with cryptography is broken; one that confuses
two flavours of PC hardware is behaving sensibly on a genuinely fuzzy boundary.

---

## The interactive demo

```bash
streamlit run app/streamlit_app.py
```

Paste any text (or pick a built-in example) and get the predicted topic, the
probability spread across all 20 topics, and the per-word contribution panel.
The demo loads the trained artifact and reuses the exact `explain_prediction`
function the tests cover — no logic duplicated between the app and the library.

---

## Running it

```bash
git clone https://github.com/diskaver-it/news-topic-classifier
cd news-topic-classifier
pip install -e ".[dev]"

python -m news_classifier.train        # downloads 20NG once (~15 MB), trains, writes reports/
pytest -q                              # 24 tests, no network required
streamlit run app/streamlit_app.py     # the demo
```

Training takes a couple of minutes, most of it the cross-validation sweep over
`C`. The download happens once and is cached under `data/raw/`.

---

## Design choices, briefly

**Why TF-IDF + logistic regression.** The honest strong baseline for topic
classification, interpretable per prediction, trains in seconds, and produces a
small artifact the demo can load instantly. Reaching for a transformer first
would be the wrong instinct to show — and would forfeit the explanations.

**Why macro-F1, not accuracy.** Accuracy weights every document equally, so a
model can coast on the easy, well-separated topics. Macro-F1 averages F1 *per
class*, so a topic the model cannot handle (talk.religion.misc) drags the score
down — which is exactly what you want to surface. `C` is tuned against macro-F1
for the same reason.

**Why the official by-date split.** 20 Newsgroups ships a train/test split where
the test posts are from a later time period. That mirrors deployment — you always
predict on text written after training — so reshuffling would leak future posts
into training and flatter the numbers.

**`sublinear_tf` + bigrams.** Word informativeness grows with frequency but not
linearly, so counts become `1 + log(count)`; bigrams let "hard drive" and "new
york" be features rather than dissolving into single words.

---

## Layout

```
src/news_classifier/
├── config.py     paths, dataset settings, topic → super-category map
├── data.py       load the official train/test split (cached, offline after first run)
├── model.py      the TF-IDF + logistic-regression pipeline
├── train.py      CV tuning, fit, the leakage experiment, persistence
├── evaluate.py   macro-F1, per-class report, top confusions
├── explain.py    per-prediction word contributions + per-topic defining words
└── plots.py      confusion matrix, per-class F1, top-features figures

app/streamlit_app.py   the interactive demo
tests/                 24 tests on a synthetic corpus — no network, ~10s
reports/               metrics.json, leakage_comparison.json, top_features_per_topic.json, figures/
```

Raw data and the trained artifact are gitignored — both are regenerated by
`python -m news_classifier.train`, and committing binaries makes diffs useless.

---

## Known limitations

- **No deep-learning baseline.** A fine-tuned transformer would likely beat this
  on raw accuracy. It would also be slower, larger, and far harder to explain —
  the trade-off is the point, not an oversight. A fair comparison would be a good
  next step.
- **English, single-label, topic-level.** The model assumes exactly one of these
  20 topics; it has no "none of the above" option and no notion of documents that
  span two topics.
- **Vocabulary is frozen at training time.** New jargon (a product or event that
  post-dates the corpus) contributes nothing until the model is retrained.
- **No calibration pass.** The probabilities are usable for ranking and for the
  demo, but have not been explicitly calibrated; treat a "99%" as "very
  confident", not as a literal frequency.

## Licence & attribution

Code: MIT. Dataset: the 20 Newsgroups collection, distributed with scikit-learn;
originally assembled by Ken Lang.
