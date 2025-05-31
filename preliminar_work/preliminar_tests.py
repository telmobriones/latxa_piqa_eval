from datasets import load_dataset

ds_eu = load_dataset("HiTZ/PIQA-eu")
ds_en = load_dataset("ybisk/piqa", trust_remote_code=True)

# load validation set
ds_eu = ds_eu["validation"]
ds_en = ds_en["validation"]

# # Save "goal", "sol1" and "sol2" column contents separated by '\t' to a .csv file
# def save_to_csv(dataset, filename):
#     with open(filename, 'w') as f:
#         for example in dataset:
#             f.write(f"{example['goal']}\t{example['sol1']}\t{example['sol2']}\n")
#     print(f"Saved dataset to {filename}")

# # Save the datasets to CSV files
# save_to_csv(ds_eu, "PIQA-eu.csv")
# save_to_csv(ds_en, "PIQA-en.csv") 

ds_en = ds_en.add_column("idx", list(range(len(ds_en))))

eu_ids = set(ds_eu["idx"])
en_ids = set(ds_en["idx"])

print(f"Length of PIQA-eu: {len(ds_eu)}")
print(f"Length of PIQA-en: {len(ds_en)}")

print(f"Number of common ids: {len(eu_ids.intersection(en_ids))}")
print(f"Number of ids not in common: {len(eu_ids.symmetric_difference(en_ids))}")
print(f"IDs in PIQA-en but not in PIQA-eu: {eu_ids.symmetric_difference(en_ids)}")

for i in eu_ids.symmetric_difference(en_ids):
    print(f"PIQA-en: {ds_en[i]}")

ds_en_aligned = ds_en.filter(lambda example: example["idx"] in eu_ids)
assert set(ds_en_aligned["idx"]) == eu_ids  # This should pass if alignment is correct