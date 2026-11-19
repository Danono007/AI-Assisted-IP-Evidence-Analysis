import json
import re
import ipaddress
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = Path("working/resources/code")
OUTPUT = Path("outputs")
CHARTS = Path("charts")

OUTPUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

IP_PATTERN = re.compile(
    r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"
)

# --------------------------------------------------
# Load original datasets
# --------------------------------------------------

def load_jsonl(filename, split):
    records = []

    with (BASE / filename).open(encoding="utf-8") as file:
        for row_number, line in enumerate(file, start=1):
            record = json.loads(line)

            records.append({
                "source_row": f"{split}:{row_number}",
                "context": record["prompt"],
                "label": record["completion"].strip().lower()
            })

    return pd.DataFrame(records)


train = load_jsonl("dataset_train.jsonl", "train")
test = load_jsonl("dataset_test.jsonl", "test")

label_map = {"negative": 0, "positive": 1}

X_train = train["context"]
X_test = test["context"]

y_train = train["label"].map(label_map).to_numpy()
y_test = test["label"].map(label_map).to_numpy()

assert len(train) == 183
assert len(test) == 46
assert not train["label"].isna().any()
assert not test["label"].isna().any()

# --------------------------------------------------
# Extract candidate and independently validate IPv4
# --------------------------------------------------

def extract_candidate(context):
    if ";" not in context:
        return ""

    candidate_field = context.split(";", 1)[1].split("->", 1)[0]
    match = IP_PATTERN.search(candidate_field)

    return match.group(0) if match else ""


def is_valid_ipv4(candidate):
    try:
        ipaddress.IPv4Address(candidate)
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


test_candidates = test["context"].apply(extract_candidate)
ipv4_predictions = np.array([
    int(is_valid_ipv4(candidate))
    for candidate in test_candidates
])

# --------------------------------------------------
# Build word + character TF-IDF model
# --------------------------------------------------

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True
        )
    )
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=3000,
        random_state=SEED
    ))
])

# Fit only on the original training records.
model.fit(X_train, y_train)

# Predict the fixed test records.
ml_predictions = model.predict(X_test)
ml_probabilities = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Baseline predictions
# --------------------------------------------------

majority_class = int(pd.Series(y_train).mode().iloc[0])
majority_predictions = np.full(len(y_test), majority_class)

# --------------------------------------------------
# Evaluation functions
# --------------------------------------------------

def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


results = [
    calculate_metrics(
        "TF-IDF Logistic Regression", y_test, ml_predictions
    ),
    calculate_metrics(
        "Majority-class baseline", y_test, majority_predictions
    ),
    calculate_metrics(
        "IPv4-validity baseline", y_test, ipv4_predictions
    )
]

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(OUTPUT / "model_comparison_metrics.csv", index=False)

# --------------------------------------------------
# Classification report and confusion matrix
# --------------------------------------------------

print("\n========== DATASET SUMMARY ==========")
print("Training records:", len(train))
print("Test records:", len(test))
print("Training positive:", int((y_train == 1).sum()))
print("Training negative:", int((y_train == 0).sum()))
print("Test positive:", int((y_test == 1).sum()))
print("Test negative:", int((y_test == 0).sum()))
print("Majority training class:", "positive" if majority_class else "negative")

print("\n========== MACHINE LEARNING RESULTS ==========")
print(metrics_df.to_string(index=False))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    ml_predictions,
    labels=[0, 1],
    target_names=["negative", "positive"],
    zero_division=0
))

print("\n========== MACHINE LEARNING CONFUSION MATRIX ==========")
ml_cm = confusion_matrix(y_test, ml_predictions, labels=[0, 1])
print(ml_cm)

tn, fp, fn, tp = ml_cm.ravel()

print("\nTrue negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

print("\n========== MACHINE LEARNING SCORE METRICS ==========")
print("AUROC:", roc_auc_score(y_test, ml_probabilities))
print("AUPRC:", average_precision_score(y_test, ml_probabilities))

# --------------------------------------------------
# Save per-record predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "source_row": test["source_row"],
    "context": test["context"],
    "candidate": test_candidates,
    "true_label": test["label"],
    "ml_prediction": [
        "positive" if value == 1 else "negative"
        for value in ml_predictions
    ],
    "positive_probability": ml_probabilities,
    "majority_prediction": [
        "positive" if value == 1 else "negative"
        for value in majority_predictions
    ],
    "ipv4_valid": test_candidates.apply(is_valid_ipv4),
    "ipv4_validity_prediction": [
        "positive" if value == 1 else "negative"
        for value in ipv4_predictions
    ]
})

predictions_df["ml_correct"] = (
    predictions_df["true_label"] == predictions_df["ml_prediction"]
)

predictions_df.to_csv(
    OUTPUT / "predictions.csv",
    index=False
)

# --------------------------------------------------
# Save confusion matrix chart
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=ml_cm,
    display_labels=["Negative", "Positive"]
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

ax.set_title("Offline Logistic Regression - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(CHARTS / "model_confusion_matrix.png", dpi=200)
plt.close()

# --------------------------------------------------
# Save model comparison chart
# --------------------------------------------------

metrics_df.set_index("model")[
    ["accuracy", "precision", "recall", "f1"]
].plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))

plt.ylabel("Score")
plt.title("Model and Baseline Performance Comparison")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(CHARTS / "model_performance_comparison.png", dpi=200)
plt.close()

# --------------------------------------------------
# Save reproducibility details
# --------------------------------------------------

with (OUTPUT / "model_run_details.txt").open("w") as file:
    file.write(f"Random seed: {SEED}\n")
    file.write("Model: TF-IDF word and character n-grams + Logistic Regression\n")
    file.write("Training data: original dataset_train.jsonl (183 records)\n")
    file.write("Test data: original dataset_test.jsonl (46 records)\n")
    file.write("Vectorizer fitted on training data only.\n")
    file.write("No external AI service or data upload used.\n")

print("\n========== OUTPUT FILES ==========")
print("outputs/model_comparison_metrics.csv")
print("outputs/predictions.csv")
print("outputs/model_run_details.txt")
print("charts/model_confusion_matrix.png")
print("charts/model_performance_comparison.png")

print("\nMODEL EVALUATION COMPLETED.")
