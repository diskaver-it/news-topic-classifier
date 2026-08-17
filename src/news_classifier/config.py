"""Paths, dataset settings and label metadata - the single source of truth.

Keeping every constant here (instead of scattering literals across scripts) is
what lets a reader change one thing - the seed, the vectoriser vocabulary size,
the list of topics - and have the whole pipeline follow.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # sklearn's 20news cache lives here

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"
LEAKAGE_REPORT_FILE = REPORTS_DIR / "leakage_comparison.json"
TOP_FEATURES_FILE = REPORTS_DIR / "top_features_per_topic.json"
CALIBRATION_REPORT_FILE = REPORTS_DIR / "calibration.json"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# The 20 Newsgroups corpus (~18k Usenet posts across 20 topics), served by
# scikit-learn. It ships with an official *by-date* train/test split: the test
# set is a later slice of time than the training set. That is a realistic split
# - a deployed classifier always predicts on posts written after it was trained
# - so we use it as given rather than reshuffling.
#
# `remove` strips the three parts of a post that leak the answer without any
# real language understanding:
#   - headers: contain the newsgroup name outright, plus tell-tale sender domains
#   - footers: signatures that identify recurring posters and their groups
#   - quotes:  quoted text from a parent post, often from the same thread/topic
# Leaving them in inflates accuracy dramatically while teaching the model to
# read metadata instead of content. train.py measures exactly how much - see
# reports/leakage_comparison.json.
REMOVE_PARTS = ("headers", "footers", "quotes")

RANDOM_STATE = 42

# Human-readable names for the 20 canonical newsgroups, in sklearn's label
# order (0..19). Grouping them by super-category is useful for the confusion
# analysis: most of the model's mistakes are *within* a super-category
# (e.g. two comp.* groups), which is a very different kind of error than
# confusing hockey with cryptography.
SUPERCATEGORIES = {
    "alt.atheism": "religion/politics",
    "comp.graphics": "computers",
    "comp.os.ms-windows.misc": "computers",
    "comp.sys.ibm.pc.hardware": "computers",
    "comp.sys.mac.hardware": "computers",
    "comp.windows.x": "computers",
    "misc.forsale": "misc",
    "rec.autos": "recreation",
    "rec.motorcycles": "recreation",
    "rec.sport.baseball": "recreation",
    "rec.sport.hockey": "recreation",
    "sci.crypt": "science",
    "sci.electronics": "science",
    "sci.med": "science",
    "sci.space": "science",
    "soc.religion.christian": "religion/politics",
    "talk.politics.guns": "religion/politics",
    "talk.politics.mideast": "religion/politics",
    "talk.politics.misc": "religion/politics",
    "talk.religion.misc": "religion/politics",
}

# --------------------------------------------------------------------------
# Model / vectoriser
# --------------------------------------------------------------------------
# TF-IDF vocabulary limits. min_df drops words appearing in fewer than N
# documents (typos, one-off tokens); max_df drops words in more than X% of docs
# (corpus-wide boilerplate the stop-word list misses). These keep the feature
# matrix sane and the model from keying on noise.
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.5
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)   # unigrams + bigrams ("new york" carries more than the two words)

# Inverse regularisation strength for the logistic-regression head. Chosen by
# the cross-validation sweep in train.py rather than hard-coded blindly.
LOGREG_C = 1.0

# How many top tokens to store per topic for the report and the demo.
TOP_FEATURES_PER_TOPIC = 15

# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
# The demo prints a probability next to every prediction, so that probability
# has to mean something. See calibration.py: a temperature is fitted on
# out-of-fold logits and stored with the model, and the calibrated confidence
# then supports an abstention rule ("I am not sure enough - send this to a
# human") instead of forcing a guess among 20 topics.
CALIBRATION_BINS = 15

# Folds used to produce the out-of-fold logits the temperature is fitted on.
# Fitting it on in-sample logits would calibrate against predictions the model
# has already seen the answers to, which is how a model ends up "calibrated"
# only on its training set.
CALIBRATION_CV_FOLDS = 3

# The operating point for abstention: answer only where the model would be
# right this often. 0.90 against a 0.695 overall accuracy is a real product
# choice - it trades coverage for a usable guarantee on what is answered.
ABSTAIN_TARGET_ACCURACY = 0.90
