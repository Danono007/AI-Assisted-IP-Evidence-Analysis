AI-Assisted IP Evidence Identification and Model Evaluation
ICDFA-DFIR-AI-026 | AI for Forensics and Capstone Intelligence A
Author: Daniel Ukpong
Status: Academic capstone / synthetic teaching-data experiment
Overview
This project evaluates an offline workflow for identifying IPv4-like candidates in synthetic mobile-device configuration text and comparing a transparent syntax-based baseline with a machine-learning classifier.

It covers evidence integrity, dataset inspection, regex candidate extraction, strict IPv4 syntax validation, offline model training and evaluation, error analysis, analyst review, and responsible reporting.
Scope and safety: This is an academic exercise using synthetic teaching artefacts. Source labels identify the intended candidate label in the supplied dataset; they do not establish routability, reachability, ownership, safety, public/private status, or maliciousness. No live incident or real device is being investigated.

Objectives
- Preserve and inventory supplied files and record SHA-256 integrity information.
- Inspect fixed train/test datasets, label distribution, duplicates, and split overlap.
- Extract IPv4-shaped candidates with a regular expression.
- Independently validate syntax using Python's ipaddress module.
- Train and evaluate a local TF-IDF + Logistic Regression classifier.
- Compare the ML model with simple baselines.
- Review representative cases and document errors, ambiguity, and escalation.
- Evaluate limitations of historical fine-tuning metrics provided with the course materials.
- Keep the analysis reproducible and offline.
Repository Contents
CIP-B107-LAB01/
├── originals/                         # Original archive (keep local; do not push)
├── working/resources/                  # Extracted course material (review before sharing)
├── scripts/
├── notebooks/                           # Optional notebooks
├── outputs/
├── charts/
├── reports/                              # Final report PDF
├── briefing/                              # Executive briefing
├── screenshots/                            # Lab evidence screenshots
└── logs/                                   # Environment, inventory, hash records
Repository hygiene: Exclude the original archive, .venv, secrets, credential-bearing historical scripts, and unreviewed extracted material from version control. Publish course-provided data only if course terms permit it. If data cannot be redistributed, document how to obtain it through the approved course channel.
Dataset Summary
The fixed supplied split was retained for the main evaluation.

| Split | Records | Positive | Negative | Unique prompts | Duplicate occurrences |
||---:|---:|---:|---:|---:|
| Training | 183 | 163 | 20 | 138 | 45 |
| Test | 46 | 34 | 12 | 25 | 21 |
| Combined | 229 | 197 | 32 | — | — |

Exact prompt overlap was found between training and test sets:
- 11 overlapping prompts
- 25 unique test prompts
- 44% of unique test prompts overlap with training
This leakage risk is disclosed rather than silently corrected, because the assignment specified the supplied split. It reduces confidence in generalization. Duplicate rows also affect row-level metrics.
Candidate Extraction and Syntax Validation
The extractor uses this regex to identify IPv4-shaped numeric sequences:
(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])
Regex matching is candidate generation only. Each candidate is separately checked using Python's strict ipaddress.IPv4Address validation.
Candidate review results
Result	Count
Total records	229
Regex candidates extracted	229
Strict IPv4 syntax valid	201
Strict IPv4 syntax invalid	28
Source-labelled positive	197
Source-labelled negative	32


The review export is outputs/candidate_review.csv. It retains source context and label alongside the candidate and validation result.
Machine-Learning Method
- Model: Logistic Regression
- Feature representation: Combined word and character TF-IDF
  - Word n-grams: 1–2
  - Character n-grams: 2–5
- Training set: 183 records
- Test set: 46 records
- Random seed: 42
- Vectorizer fitting: Training data only
- Maximum iterations: 3,000
- Execution: Local/offline; no third-party AI service or data upload
The supplied split was not modified for the main experiment.
Model Results
Model	Accuracy	Precision	Recall	F1
TF-IDF Logistic Regression	82.61%	80.95%	100.00%	89.47%
Majority-class baseline	73.91%	73.91%	100.00%	85.00%
IPv4-validity baseline	95.65%	94.44%	100.00%	97.14%


Logistic Regression confusion matrix
Class order: Negative, Positive
	Predicted Negative	Predicted Positive
Actual Negative	4 (TN)	8 (FP)
Actual Positive	0 (FN)	34 (TP)


- AUROC: 0.99265
- AUPRC: 0.99740
- Negative-class recall: 33.33%
- Positive-class recall: 100% on this supplied test set
Despite high ranking metrics, the classifier produced eight false positives at the evaluated threshold. The syntax baseline performed better on this dataset. This does not prove strict IPv4 validation is universally sufficient for forensic classification.
Error Analysis and Analyst Review
Logistic Regression false positives were source-labelled negatives. Most involved malformed octets such as 256 or 257; two involved 0.0.0.0, syntactically valid but contextually meaningful as an unspecified address.

An eight-record analyst review was conducted with four source-labelled positives and four negatives. Five cases were marked for contextual verification or escalation. Escalation means additional review is needed; it is not a finding of malicious activity.
Class-Imbalance Demonstration
The test set contains 34 positives and 12 negatives. A classifier that predicts positive for every test record obtains:
- Accuracy: 73.91%
- Negative recall: 0%
- Negative records correctly detected: 0 of 12
This demonstrates why accuracy alone is insufficient with imbalanced data. Class-specific metrics and error review are essential.
Historical Fine-Tuning Artefact Review
Course-provided historical logs were analyzed separately from the new offline model:
- fine_tuning_steps.txt: 734 records
- result.csv: 733 records
- result.csv contains 92 non-null validation metric observations and four logged observations for several classification metrics.
These historical metrics are not the results of the newly trained Logistic Regression classifier. Historical scripts using older API patterns were inspected only and were not executed. No data was uploaded to a remote model service.
Reproducibility
Developed in Kali Linux using Python in a virtual environment. Package versions are recorded in logs/environment_versions.txt.

From the project root:
cd ~/CIP-B107-LAB01
source .venv/bin/activate
Re-run analysis scripts in sequence if you intend to regenerate outputs. Verify each output before committing:
python scripts/create_evidence_manifest.py
python scripts/inspect_historical_metrics.py
python scripts/build_candidate_review.py
python scripts/train_evaluate_model.py
python scripts/prepare_analyst_review.py
python scripts/complete_analyst_review.py
python scripts/imbalance_counterexample.py
This sequence reflects the project scripts. The course-supplied legacy API scripts are not part of it.
Evidence and Report
Add the report and screenshots after review. Suggested artifacts:
- reports/ICDFA_DFIR_AI_Evidence_Identification_Report.pdf
- briefing/Executive_Briefing.pdf (or slide deck, if required)
- Screenshots under screenshots/, referenced by evidence IDs in the report.

Key Limitations
1. Synthetic teaching data does not establish real-world forensic reliability.
2. Eleven exact prompts overlap across the fixed train/test split.
3. Duplicate records can inflate the influence of repeated examples.
4. Positive examples dominate the dataset.
5. IP-shaped strings may appear in versions, URLs, paths, placeholders, or malformed values.
6. The classifier produced eight false-positive errors at the evaluated threshold.
7. No live reachability, attribution, geolocation, or network verification was performed.
8. Dataset labels do not establish maliciousness or incident relevance.
Conclusion
The project implemented a reproducible offline pipeline for candidate extraction, syntax validation, model evaluation, baseline comparison, and analyst review. On this dataset, the IPv4-validity baseline outperformed Logistic Regression. The ML classifier had perfect positive recall on the supplied test set but weak negative-class recall and eight false positives.

Takeaway: Automated extraction and classification are triage aids—not final forensic decisions. Preserve original context, validate syntax independently, review ambiguous cases, disclose leakage and limitations, and keep the analyst accountable.
Author
Daniel Ukpong
Cybersecurity | Digital Forensics | Cloud & DevOps
Disclaimer
This repository is for educational and portfolio purposes. It is not intended to be an automated incident-response, threat-attribution, or evidence-disposition system.
