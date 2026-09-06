"""Interactive demo for the 20 Newsgroups topic classifier.

Run with:  streamlit run app/streamlit_app.py

Paste any text; the app shows the predicted topic, the spread across all 20,
and which words in that text drove the decision - the last panel only being
possible because the model is linear.

The confidence is the calibrated one, and below the abstention threshold the
app says it is not sure. A demo that produces a confident-looking topic for any
input misrepresents the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Make the src/ package importable whether the app is launched from the repo
# root or from app/ - Streamlit does not install the package for you.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_classifier import config, explain  # noqa: E402

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="wide")

EXAMPLES = {
    "— pick an example —": "",
    "Space / NASA": (
        "The spacecraft entered orbit around Mars after a seven-month journey. "
        "NASA engineers confirmed the propulsion burn went nominally and the "
        "probe will begin mapping the surface next week."
    ),
    "Hockey": (
        "What a game last night - the goalie stopped 45 shots and they still "
        "took it to overtime. If they keep this up in the playoffs the whole "
        "division is in trouble. That power play is unstoppable."
    ),
    "Cryptography / privacy": (
        "The proposed key-escrow system would give the government access to "
        "every encrypted message. The clipper chip is a backdoor by another "
        "name, and no amount of oversight makes weak encryption safe."
    ),
    "PC hardware": (
        "I swapped the motherboard but now the machine won't POST. The RAM is "
        "seated, the CPU fan spins, but I get no video. Could it be the power "
        "supply, or did I fry something installing the new card?"
    ),
    "None of the 20 topics": (
        "The sourdough needs a longer autolyse if the flour is high-protein. "
        "I fold it three times over four hours, then retard the dough overnight "
        "in the fridge before baking it in a preheated dutch oven."
    ),
}


@st.cache_resource
def load_artifact():
    """Load the trained model once and cache it across reruns.

    `cache_resource` is the right cache for an unserialisable, expensive object
    like a fitted pipeline - Streamlit keeps a single instance for the whole
    session instead of reloading it on every keystroke.
    """
    import joblib

    if not config.MODEL_FILE.exists():
        return None
    return joblib.load(config.MODEL_FILE)


def probability_chart(probabilities: dict) -> alt.Chart:
    """Horizontal bar chart of the top topics by probability."""
    frame = (
        pd.DataFrame({"topic": list(probabilities), "probability": list(probabilities.values())})
        .sort_values("probability", ascending=False)
        .head(8)
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", axis=alt.Axis(format="%"), title="Probability"),
            y=alt.Y("topic:N", sort="-x", title=None),
            color=alt.Color("probability:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[alt.Tooltip("topic:N"), alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=280)
    )


def contribution_chart(tokens: list) -> alt.Chart:
    """Signed contribution of each token toward the predicted topic."""
    frame = pd.DataFrame(tokens)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution to the predicted topic"),
            y=alt.Y("token:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.contribution > 0,
                alt.value("#2c7fb8"),      # pushed toward the topic
                alt.value("#d95f0e"),      # pushed away
            ),
            tooltip=[alt.Tooltip("token:N"), alt.Tooltip("contribution:Q", format=".3f")],
        )
        .properties(height=280)
    )


def main() -> None:
    st.title("📰 News Topic Classifier")
    st.caption(
        "A TF-IDF + logistic-regression model trained on 20 Newsgroups. "
        "Paste text to see the predicted topic — and which words drove it."
    )

    artifact = load_artifact()
    if artifact is None:
        st.error(
            "No trained model found. Run `python -m news_classifier.train` first "
            "to create models/model.joblib."
        )
        st.stop()

    pipeline = artifact["pipeline"]
    target_names = artifact["target_names"]
    # Older artifacts predate calibration; 1.0 and no threshold reproduce the
    # previous behaviour rather than crashing the demo.
    temperature = artifact.get("temperature", 1.0)
    abstain_threshold = artifact.get("abstain_threshold")

    with st.sidebar:
        st.header("About")
        about = f"""
- **{artifact['n_train_docs']:,}** training posts, **{len(target_names)}** topics
- Model: TF-IDF (1–2 grams) → logistic regression, `C={artifact['best_C']}`
- Trained: {artifact['trained_at'][:10]}
- scikit-learn {artifact['sklearn_version']}

The **word contributions** panel is possible because the model is linear:
each word's push toward a topic is its TF-IDF value times the model weight.
"""
        if abstain_threshold:
            about += (
                f"\nConfidence is temperature-scaled (`T={temperature:.2f}`), so a "
                f"percentage shown here is one the reports measured. Below "
                f"**{abstain_threshold:.0%}** the app says it is not sure instead "
                f"of guessing.\n"
            )
        st.markdown(about)
        st.markdown("[Source & write-up](https://github.com/diskaver-it/news-topic-classifier)")

    choice = st.selectbox("Load an example, or write your own below:", list(EXAMPLES))
    text = st.text_area(
        "Text to classify",
        value=EXAMPLES[choice],
        height=200,
        placeholder="Paste a news post, forum message or article here...",
    )

    if not st.button("Classify", type="primary"):
        return

    if not text.strip():
        st.warning("Enter some text first.")
        return

    result = explain.explain_prediction(
        pipeline, text, target_names, temperature=temperature
    )
    abstains = bool(abstain_threshold and result["confidence"] < abstain_threshold)

    top, right = st.columns([1, 1])
    with top:
        st.subheader("Prediction")
        if abstains:
            st.warning(
                f"**Not confident enough to answer.** The best guess is "
                f"`{result['predicted_topic']}` at {result['confidence']:.1%}, "
                f"below the {abstain_threshold:.0%} cutoff — this text may not "
                f"belong to any of the 20 newsgroups, or it may sit between two "
                f"of them. The ranking below is still shown; it is just not "
                f"worth acting on."
            )
        else:
            st.metric(result["predicted_topic"], f"{result['confidence']:.1%} confident")
        st.caption(
            f"Runner-up: **{result['runner_up_topic']}** "
            f"({result['runner_up_confidence']:.1%})"
        )
        st.altair_chart(probability_chart(result["probabilities"]), use_container_width=True)

    with right:
        st.subheader("Why this topic?")
        st.caption("Words in your text ranked by how hard they pushed the decision.")
        st.altair_chart(
            contribution_chart(result["top_contributing_tokens"]), use_container_width=True
        )


if __name__ == "__main__":
    main()
