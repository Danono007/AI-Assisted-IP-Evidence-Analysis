import pandas as pd
from pathlib import Path

path = Path("outputs/analyst_review.csv")
df = pd.read_csv(path, dtype={"source_row": str})

assessments = {
    "test:1": (
        "The candidate contains octet 608, exceeding the IPv4 range of 0-255. It is not syntactically valid IPv4. The model correctly predicted negative.",
        "Reject as invalid IPv4 candidate",
        "No"
    ),
    "test:20": (
        "The candidate contains octet 257, exceeding 255. The source label is negative, but the model predicted positive. This is a model false positive.",
        "Reject as invalid IPv4 candidate; record model error",
        "Yes"
    ),
    "test:22": (
        "The candidate contains octet 256, exceeding 255. The device-registration context does not make it a valid IPv4 address. The model incorrectly predicted positive.",
        "Reject as invalid IPv4 candidate; record model error",
        "Yes"
    ),
    "test:44": (
        "0.0.0.0 passes IPv4 syntax validation, but it is the unspecified address, not an ordinary host endpoint. The source label is negative while the model predicted positive. Contextual review is needed.",
        "Flag for contextual review; retain source negative label",
        "Yes"
    ),
    "test:2": (
        "The candidate 2.20.31.4 is syntactically valid IPv4. The source label and model prediction are both positive. No contradiction is apparent in the supplied context.",
        "Retain as source-labelled positive",
        "No"
    ),
    "test:3": (
        "The candidate 13.3.1.17 is syntactically valid IPv4. The original text appends D50 immediately after it, so the surrounding string should be reviewed before treating the candidate as a confirmed network address.",
        "Retain candidate; verify surrounding context",
        "Yes"
    ),
    "test:5": (
        "The candidate 104.36.225.166 is syntactically valid IPv4. The source label and model prediction are both positive, and the displayed context contains the candidate directly.",
        "Retain as source-labelled positive",
        "No"
    ),
    "test:9": (
        "The candidate 10.5.0.1 is syntactically valid IPv4, but it appears inside a software JavaScript resource path. It could represent a software version component rather than a network endpoint. Retain the source annotation while flagging the context for verification.",
        "Retain candidate; verify whether context is an IP or version string",
        "Yes"
    )
}

# Explicitly create text columns to avoid Pandas float64 assignment errors.
for column in [
    "analyst_assessment",
    "review_disposition",
    "escalation_required"
]:
    df[column] = df[column].astype("object")

if set(df["source_row"]) != set(assessments):
    raise ValueError("The review CSV does not contain the expected eight source rows.")

for index, row in df.iterrows():
    assessment, disposition, escalation = assessments[row["source_row"]]
    df.at[index, "analyst_assessment"] = assessment
    df.at[index, "review_disposition"] = disposition
    df.at[index, "escalation_required"] = escalation

df.to_csv(path, index=False)

print("\n========== COMPLETED ANALYST REVIEW ==========")
print("Records reviewed:", len(df))
print("Positive source labels:", (df["true_label"] == "positive").sum())
print("Negative source labels:", (df["true_label"] == "negative").sum())
print("Escalation required:", (df["escalation_required"] == "Yes").sum())
print("Review file:", path)

print("\n========== REVIEW DISPOSITIONS ==========")
print(
    df[
        ["source_row", "candidate", "review_disposition",
         "escalation_required"]
    ].to_string(index=False)
)

print("\nAll eight analyst assessments have been populated.")
