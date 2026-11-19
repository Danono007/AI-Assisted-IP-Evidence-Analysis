import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone

root = Path(".")
source = root / "working" / "resources"
output = root / "outputs" / "evidence_manifest.csv"

output.parent.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = sorted(path for path in source.rglob("*") if path.is_file())

with output.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=[
            "relative_path",
            "file_size_bytes",
            "sha256",
            "collection_recorded_utc"
        ]
    )
    writer.writeheader()

    for path in files:
        writer.writerow({
            "relative_path": path.as_posix(),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "collection_recorded_utc": datetime.now(
                timezone.utc
            ).isoformat()
        })

print("Files recorded:", len(files))
print("Manifest:", output)
print("Manifest SHA-256:", sha256_file(output))
