import os
import requests
import pandas as pd
from datasets import load_dataset

VALID_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
VALID_LABELS_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"

dataset_dir = "PIQA_dataset"
os.makedirs(dataset_dir, exist_ok=True)

for url, fname in [(VALID_URL, "valid.jsonl"), (VALID_LABELS_URL, "valid-labels.lst")]:
    response = requests.get(url)
    response.raise_for_status()  # ensure we stop on download errors
    with open(os.path.join(dataset_dir, fname), "wb") as f:
        f.write(response.content)

ds = load_dataset(
    "json",
    data_files={"validation": os.path.join(dataset_dir, "valid.jsonl")},
    split="validation"
)

labels = pd.read_csv(
    os.path.join(dataset_dir, "valid-labels.lst"),
    header=None,
    names=["label"]
)["label"].tolist()

ds = ds.add_column("label", labels)

out_path = os.path.join(dataset_dir, "PIQA_dataset.jsonl")
ds.to_json(out_path, orient="records", lines=True)
print(f"Saved combined dataset to {out_path}")

print("\nFirst 5 examples:")
for i, example in enumerate(ds.select(range(5))):
    print(f"Example {i}:", example)
print(f"\nTotal examples: {len(ds)}")
