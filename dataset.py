from datasets import load_dataset

dataset_name = (
    "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
)

ds = load_dataset(dataset_name, split="train", streaming=True)

print("\n--- Pulling First 3 Samples ---")
for i, sample in enumerate(ds):
  print(f"\n[Sample {i + 1}] Length of text: {len(sample['text'])} characters")
  print(f"Snippet: {sample['text'][:300]}...\n")
  if i >= 2:
    break
