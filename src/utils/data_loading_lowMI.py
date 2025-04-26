# %%
import json
import re
import shutil
import zipfile
from pathlib import Path

import torch as pt
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from datasets import (
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)


def load_wmdp_deduped(paths, task="wmdp-deduped"):
    base_url = "https://raw.githubusercontent.com/aghyad-deeb/unlearning_evaluation/refs/heads/main/data"
    return load_dataset(
        "json",
        data_files=[f"{base_url}/{task}/{path}.jsonl" for path in paths],
        split="train",
    )


attack_mcq = load_wmdp_deduped(["split_0", "split_1", "split_2", "split_3"])
eval_mcq = load_wmdp_deduped(["split_4"])
attack_corpus = load_wmdp_deduped(["corpus_split_0", "corpus_split_1", "corpus_split_2", "corpus_split_3"])  # fmt: skip
eval_corpus = load_wmdp_deduped(["corpus_split_4"])

# %%
model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

unlearning_corpus = concatenate_datasets([attack_corpus, eval_corpus])
unlearning_corpus = unlearning_corpus.shuffle(seed=42)
unlearning_corpus = unlearning_corpus.batch(8)
unlearning_batches = [
    tokenizer(
        x["text"],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    for x in unlearning_corpus
]

# %%
len(unlearning_batches)