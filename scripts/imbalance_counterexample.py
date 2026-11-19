import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

source = Path("outputs/predictions.csv")
destination = Path("outputs/imbalance_counterexample.txt")

df = pd.read_csv(source)

actual = df["true_label"]
majority_prediction = pd.Series(
    ["positive"] * len(df),
    index=df.index
)

labels = ["negative", "positive"]

matrix = confusion_matrix(
    actual,
    majority_prediction,
    labels=labels
)

report = classification_report(
    actual,
    majority_prediction,
    labels=labels,
    zero_division=0
)

accuracy = accuracy_score(actual, majority_prediction)

lines = [
    "AI-ASSISTED FORENSIC EVALUATION",
    "CLASS IMBALANCE COUNTEREXAMPLE",
    "=" * 55,
    "",
    f"Test records: {len(df)}",
    f"Actual positive records: {(actual == 'positive').sum()}",
    f"Actual negative records: {(actual == 'negative').sum()}",
    "",
    "Baseline strategy: predict every record as positive.",
    f"Baseline accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
    "",
    "Confusion matrix (actual rows; predicted columns)",
    "Class order: negative, positive",
    str(matrix),
    "",
    "Classification report:",
    report,
    "",
    "INTERPRETATION",
    "-" * 55,
    "The baseline achieves 73.91% accuracy by predicting the",
    "majority class for every test record.",
    "It correctly identifies all positive records but fails to",
    "identify any negative records.",
    "Negative-class recall is therefore 0%.",
    "",
    "This demonstrates why accuracy alone is insufficient for",
    "evaluating an imbalanced forensic classification dataset.",
    "",
    "The counterexample is a baseline illustration, not a new",
    "model-training experiment."
]

text = "\n".join(lines)
print(text)

destination.write_text(text + "\n", encoding="utf-8")
print("\nSaved:", destination)
