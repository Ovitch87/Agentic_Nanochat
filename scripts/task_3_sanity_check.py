import os
from pprint import pprint

os.environ["NANOCHAT_BASE_DIR"] = os.path.abspath("task_2_results")

from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk

datasets = [
    ("MMLU", MMLU(subset="all", split="auxiliary_train")),
    ("GSM8K", GSM8K(subset="main", split="train")),
    ("SmolTalk", SmolTalk(split="train")),
]

for name, dataset in datasets:
    print(f"\n{name}: {len(dataset):,} examples")

    for i in range(2):
        print(f"\nExample {i + 1}:")
        pprint(dataset[i])