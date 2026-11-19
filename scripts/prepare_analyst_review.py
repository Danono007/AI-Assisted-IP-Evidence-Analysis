import pandas as pd
from pathlib import Path

source = Path("outputs/predictions.csv")
destination = Path("outputs/analyst_review.csv")

df = pd.read_csv(source)

# Deliberately selected distinct test contexts.
selected_rows = [
    "test:1",
    "test:20",
    "test:22",
    "test:44",
    "test:2",
    "test:3",
    "test:5",
    "test:9",
]

review = df[df["source_row"].isin(selected_rows)].copy()

# Preserve the intended order.
review["review_order"] = review["source_row"].map(
    {row: index for index, row in enumerate(selected_rows)}
)
review = review.sort_values("review_order").drop(columns="review_order")

review["analyst_assessment"] = ""
review["review_disposition"] = ""
review["escalation_required"] = ""

columns = [
    "source_row",
    "context",
    "candidate",
    "true_label",
    "ml_prediction",
    "ipv4_valid",
    "positive_probability",
    "analyst_assessment",
    "review_disposition",
    "escalation_required",
]

review[columns].to_csv(destination, index=False)

print("\n========== DISTINCT ANALYST REVIEW SAMPLE ==========")
print("Records selected:", len(review))
print("Positive:", (review["true_label"] == "positive").sum())
print("Negative:", (review["true_label"] == "negative").sum())
print("Unique contexts:", review["context"].nunique())

print(
    review[
        ["source_row", "candidate", "true_label",
         "ml_prediction", "ipv4_valid"]
    ].to_string(index=False)
)

print("\nSaved:", destination)
