import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("working/resources/code")
out = Path("outputs")
charts = Path("charts")

out.mkdir(exist_ok=True)
charts.mkdir(exist_ok=True)

steps = pd.read_csv(base / "fine_tuning_steps.txt")
results = pd.read_csv(base / "result.csv")

for df in (steps, results):
    for col in df.columns:
        if col != "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== HISTORICAL FILE SUMMARY ==========")
print("Fine-tuning steps rows:", len(steps))
print("Result CSV rows:", len(results))

summary = []

for filename, df in [
    ("fine_tuning_steps.txt", steps),
    ("result.csv", results)
]:
    print(f"\n========== {filename} ==========")

    for col in df.columns:
        if col == "step":
            continue

        values = df[col].dropna()

        if values.empty:
            print(f"{col}: NO RECORDED VALUES")
            summary.append({
                "file": filename,
                "metric": col,
                "non_null_count": 0,
                "minimum": None,
                "maximum": None,
                "last_value": None,
                "last_value_step": None
            })
        else:
            last_index = values.index[-1]
            last_value = df.loc[last_index, col]
            last_step = df.loc[last_index, "step"]

            print(
                f"{col}: count={len(values)}, "
                f"min={values.min()}, max={values.max()}, "
                f"last={last_value} at step {last_step}"
            )

            summary.append({
                "file": filename,
                "metric": col,
                "non_null_count": len(values),
                "minimum": values.min(),
                "maximum": values.max(),
                "last_value": last_value,
                "last_value_step": last_step
            })

pd.DataFrame(summary).to_csv(
    out / "historical_metrics_summary.csv", index=False
)

# Historical training loss
plt.figure(figsize=(10, 5))
plt.plot(steps["step"], steps["training_loss"])
plt.xlabel("Step")
plt.ylabel("Training loss")
plt.title("Historical Fine-Tuning Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(charts / "historical_training_loss.png", dpi=200)
plt.close()

# Historical training sequence accuracy
plt.figure(figsize=(10, 5))
plt.plot(
    steps["step"],
    steps["training_sequence_accuracy"]
)
plt.xlabel("Step")
plt.ylabel("Training sequence accuracy")
plt.title("Historical Training Sequence Accuracy")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    charts / "historical_training_sequence_accuracy.png",
    dpi=200
)
plt.close()

# Historical training and validation loss from result.csv
plt.figure(figsize=(10, 5))
plt.plot(results["step"], results["training_loss"], label="Training loss")

if results["validation_loss"].notna().any():
    plt.plot(
        results["step"],
        results["validation_loss"],
        label="Validation loss"
    )

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Historical Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(charts / "historical_result_loss.png", dpi=200)
plt.close()

# Classification metric chart only if actual values exist
classification_columns = [
    "classification/accuracy",
    "classification/precision",
    "classification/recall",
    "classification/auroc",
    "classification/auprc",
    "classification/f1.0"
]

available = [
    col for col in classification_columns
    if col in results.columns and results[col].notna().any()
]

if available:
    plt.figure(figsize=(10, 5))
    for col in available:
        plt.plot(results["step"], results[col], label=col)

    plt.xlabel("Step")
    plt.ylabel("Metric value")
    plt.title("Historical Classification Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(charts / "historical_classification_metrics.png", dpi=200)
    plt.close()
else:
    print("\nNo non-null historical classification metrics were available.")
    print("No classification-metric chart was generated.")

print("\nSummary saved:", out / "historical_metrics_summary.csv")
print("\nGenerated chart files:")
for path in sorted(charts.glob("historical_*.png")):
    print(path)
