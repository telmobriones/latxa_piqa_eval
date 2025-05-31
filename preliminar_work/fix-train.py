import json

LABEL_PATH = "PIQA-train/train-labels.lst"
INSTACES_PATH = "PIQA-train/train.jsonl"
OUTPUT_PATH = "PIQA-train/train-predictions.jsonl"

def merge_alligned_instances_and_labels(instances_path, labels_path, output_path):
    """
    Merges instances and labels into a single JSONL file.
    """
    with open(instances_path, "r", encoding="utf-8") as fin, \
         open(labels_path, "r", encoding="utf-8") as flab, \
         open(output_path, "w", encoding="utf-8") as fout:
        for instance_line, label_line in zip(fin, flab):
            instance = json.loads(instance_line)
            label = int(label_line.strip())
            instance["label"] = label
            fout.write(json.dumps(instance, ensure_ascii=False) + "\n")
    print(f"Merged {instances_path} and {labels_path} into {output_path}", flush=True)


def main():
    merge_alligned_instances_and_labels(INSTACES_PATH, LABEL_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()