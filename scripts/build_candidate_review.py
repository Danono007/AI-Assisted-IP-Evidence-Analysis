import json
import re
import ipaddress
import pandas as pd
from pathlib import Path

BASE = Path("working/resources/code")
OUTPUT = Path("outputs/candidate_review.csv")

# Four numeric octets, allowing leading zeros for later strict validation.
IP_PATTERN = re.compile(
    r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"
)

records = []

for split_name, filename in [
    ("train", "dataset_train.jsonl"),
    ("test", "dataset_test.jsonl")
]:
    path = BASE / filename

    with path.open("r", encoding="utf-8") as file:
        for row_number, line in enumerate(file, start=1):
            record = json.loads(line)

            context = record["prompt"]
            source_label = record["completion"].strip().lower()

            # Isolate the candidate field after the semicolon.
            candidate_field = (
                context.split(";", 1)[1].split("->", 1)[0]
                if ";" in context
                else ""
            )

            match = IP_PATTERN.search(candidate_field)
            candidate = match.group(0) if match else ""

            ipv4_valid = False
            analyst_note = ""

            if match:
                try:
                    ipaddress.IPv4Address(candidate)
                    ipv4_valid = True
                    analyst_note = (
                        "Regex candidate extracted; strict IPv4 validation passed."
                    )
                except ipaddress.AddressValueError as error:
                    analyst_note = (
                        "Regex candidate extracted; strict IPv4 validation failed: "
                        + str(error)
                    )
            else:
                analyst_note = (
                    "No four-part numeric candidate found in candidate field."
                )

            records.append({
                "source_row": f"{split_name}:{row_number}",
                "context": context,
                "candidate": candidate,
                "regex_match": bool(match),
                "ipv4_valid": ipv4_valid,
                "source_label": source_label,
                "analyst_note": analyst_note
            })

columns = [
    "source_row",
    "context",
    "candidate",
    "regex_match",
    "ipv4_valid",
    "source_label",
    "analyst_note"
]

df = pd.DataFrame(records, columns=columns)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n========== CANDIDATE EXTRACTION SUMMARY ==========")
print("Total records processed:", len(df))
print("Training records:", df["source_row"].str.startswith("train:").sum())
print("Test records:", df["source_row"].str.startswith("test:").sum())
print("Regex candidates found:", df["regex_match"].sum())
print("Strict IPv4 valid:", df["ipv4_valid"].sum())
print("Strict IPv4 invalid:", (~df["ipv4_valid"]).sum())
print("Records without candidate:", (~df["regex_match"]).sum())

print("\n========== SOURCE LABEL DISTRIBUTION ==========")
print(df["source_label"].value_counts().to_string())

print("\n========== LABEL VS IPV4 VALIDITY ==========")
print(pd.crosstab(
    df["source_label"],
    df["ipv4_valid"],
    rownames=["Source label"],
    colnames=["Strict IPv4 valid"]
).to_string())

print("\n========== EXAMPLE INVALID CANDIDATES ==========")
invalid = df[df["regex_match"] & ~df["ipv4_valid"]]

for _, row in invalid.head(10).iterrows():
    print(
        row["source_row"],
        "|",
        row["candidate"],
        "|",
        row["source_label"],
        "|",
        row["analyst_note"]
    )

print("\nCSV saved to:", OUTPUT)
print("CSV columns:", ", ".join(df.columns))
