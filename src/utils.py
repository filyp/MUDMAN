import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch as pt
from datasets import IterableDataset, IterableDatasetDict, load_dataset



def load_one_oscar_shard(lang, tokenizer):
    context_len = 100
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    raw_dataset = load_dataset(
        "text",
        split="train",  # train is the only split in oscar
        data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
    )

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter1, quarter2 = half1.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            unlearn=IterableDataset.from_generator(lambda: (ex for ex in quarter1)),
            relearn=IterableDataset.from_generator(lambda: (ex for ex in quarter2)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            test=IterableDataset.from_generator(lambda: (ex for ex in quarter4)),
        )
        # process the raw data, following OSCAR-2301.py
        .map(lambda ex: {"text": json.loads(ex["text"])["content"]})
        # tokenize
        .map(
            lambda ex: tokenizer(
                ex["text"],
                return_tensors="pt",
                max_length=context_len,
                truncation=True,
            ),
        )
        # filter out the short ones
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["unlearn"]))["text"]
    return dataset


def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)


class DefaultNamespace(SimpleNamespace):
    def __getattr__(self, name):
        # This is called when an attribute doesn't exist
        return pt.tensor(pt.nan)


def get_repo_root():
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def remove_lora(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        if "lora" in key:
            del state_dict[key]
            continue
        value = state_dict[key]
        del state_dict[key]
        new_key = key.replace("base_layer.", "")
        state_dict[new_key] = value
