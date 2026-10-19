
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "learner.joblib"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

vectorizer = HashingVectorizer(
    n_features=2**12,
    alternate_sign=False,
    ngram_range=(1, 2),
)

if MODEL_PATH.exists():
    state = joblib.load(MODEL_PATH)
else:
    state = {
        "classifier": SGDClassifier(
            loss="log_loss",
            random_state=42,
        ),
        "samples": 0,
        "labels_seen": set(),
    }


def learn(question, answer, helpful):
    # Learn from both the question and the response.
    text = f"Question: {question}\nAnswer: {answer}"
    features = vectorizer.transform([text])
    label = 1 if helpful else 0

    classifier = state["classifier"]

    if state["samples"] == 0:
        classifier.partial_fit(
            features,
            [label],
            classes=[0, 1],
        )
    else:
        classifier.partial_fit(features, [label])

    state["samples"] += 1
    state["labels_seen"].add(label)

    joblib.dump(state, MODEL_PATH)

    return state["samples"]


def predict_helpfulness(question, answer):
    # Avoid making predictions with too little experience.
    if state["samples"] < 10:
        return None

    if state["labels_seen"] != {0, 1}:
        return None

    text = f"Question: {question}\nAnswer: {answer}"
    features = vectorizer.transform([text])

    probabilities = state["classifier"].predict_proba(features)[0]
    positive_index = list(state["classifier"].classes_).index(1)

    return float(probabilities[positive_index])


def get_stats():
    return {
        "training_examples": state["samples"],
        "positive_feedback": sum(state["labels_seen"]) if False else None,
        "feedback_classes_seen": sorted(state["labels_seen"]),
    }
