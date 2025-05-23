import json
import re
import shutil
import zipfile
from pathlib import Path

import torch as pt
from huggingface_hub import hf_hub_download

from datasets import IterableDataset, IterableDatasetDict, load_dataset


def looping_iter(iterable):
    # like itertools.cycle, but will not eat memory by storing element copies
    while True:
        yield from iterable


def get_batch(iter, n):
    return pt.cat([next(iter)["input_ids"] for _ in range(n)])


def prepare_dataset(raw_dataset, tokenizer, preprocess_fn=lambda ex: {}):
    # preprocess_fn is used to add additional fields to the dataset before tokenization
    context_len = 100

    # split into 4 quarters
    half1, half2 = raw_dataset.train_test_split(test_size=0.5, seed=42).values()
    quarter3, quarter4 = half2.train_test_split(test_size=0.5, seed=42).values()

    dataset = (
        # define splits; make it iterable so that it can be processed on demand
        IterableDatasetDict(
            train=IterableDataset.from_generator(lambda: (ex for ex in half1)),
            validation=IterableDataset.from_generator(lambda: (ex for ex in quarter3)),
            test=IterableDataset.from_generator(lambda: (ex for ex in quarter4)),
        ).map(preprocess_fn)
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
        # this together with truncation ensures that each example has exactly 100 tokens
        .filter(lambda ex: ex["input_ids"].shape[-1] >= context_len)
    )
    assert next(iter(dataset["test"]))["text"] != next(iter(dataset["train"]))["text"]
    return dataset


def load_one_oscar_shard(lang, tokenizer):
    # only use one ~600MB shard
    # also, streaming would make splitting too slow
    return prepare_dataset(
        load_dataset(
            "text",
            split="train",  # train is the only split in oscar
            data_files=f"hf://datasets/oscar-corpus/OSCAR-2301/{lang}_meta/{lang}_meta_part_1.jsonl.zst",
        ),
        tokenizer,
        # process the raw data, following OSCAR-2301.py
        lambda ex: {"text": json.loads(ex["text"])["content"]},
    )


def load_wikitext(tokenizer):
    return prepare_dataset(
        # train split is big enough so just use it - it's simpler
        load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"),
        tokenizer,
    )


def load_cruelty(tokenizer):
    beavertails = load_dataset("PKU-Alignment/BeaverTails")
    split = beavertails["330k_train"]
    category = "animal_abuse"
    # from 300k examples filters down to 3k
    beaver_category = split.filter(lambda ex: ex["category"][category])
    # prepare dataset further filters out short examples, down to 1.5k, or 90 batches of 16
    return prepare_dataset(
        beaver_category,
        tokenizer,
        lambda ex: {"text": ex["response"]},
        # lambda ex: {"text": ex["prompt"] + "\n" + ex["response"]},
    )


def load_beaver_safe(tokenizer):
    beavertails = load_dataset("PKU-Alignment/BeaverTails")
    split = beavertails["330k_train"]
    # from 300k examples filters down to 134k
    safe_examples = split.filter(lambda ex: ex["is_safe"])
    # prepare dataset further filters out short examples, down to 40k examples
    return prepare_dataset(
        safe_examples,
        tokenizer,
        lambda ex: {"text": ex["response"]},
        # lambda ex: {"text": ex["prompt"] + "\n" + ex["response"]},
    )


def _remove_comments_and_docstrings(code: str) -> str:
    # Remove docstrings
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # Remove single-line comments
    code = re.sub(r"#.*", "", code)
    # Remove blank lines
    code = re.sub(r"\n\s*\n", "\n", code)
    # Remove leading/trailing whitespace
    code = code.strip()
    return code


def load_python_dataset(tokenizer):
    return prepare_dataset(
        load_dataset("Nan-Do/code-search-net-python", split="train"),
        tokenizer,
        lambda ex: {"text": _remove_comments_and_docstrings(ex["code"])},
    )


def load_pile_bio_forget(tokenizer):
    pile_bio = load_dataset("lapisrocks/pile-bio", split="train")
    pile_bio_forget = pile_bio.filter(lambda ex: ex["label"])
    return prepare_dataset(
        pile_bio_forget, tokenizer, lambda ex: {"text": ex["txt_chunk"]}
    )


def load_pile_bio_retain(tokenizer):
    pile_bio = load_dataset("lapisrocks/pile-bio", split="train")
    pile_bio_retain = pile_bio.filter(lambda ex: not ex["label"])
    return prepare_dataset(
        pile_bio_retain, tokenizer, lambda ex: {"text": ex["txt_chunk"]}
    )


# # topics = dataset.unique("topic;")
# # as estimated by Claude 3.7 Sonnet in relation to bio-terrorism:
camel_topic_risk_ratings = {
    "Microbiology": 5,
    "Biotechnology": 5,
    "Virology": 5,
    "Genetics": 5,
    "Biochemistry": 4,
    "Immunology": 3,
    "Mycology": 3,
    "Parasitology": 3,
    "Cell biology": 3,
    "Biostatistics": 2,
    "Physiology": 2,
    "Entomology": 2,
    "Evolution": 2,
    "Ecology": 2,
    "Biophysics": 2,
    "Zoology": 1,
    "Neurobiology": 1,
    "Taxonomy": 1,
    "Anatomy": 1,
    "Biomechanics": 1,
    "Paleontology": 1,
    "Botany": 1,
    "Marine biology": 1,
    "Endocrinology": 1,
    "Biogeography": 1,
}


def _load_camel_bio_topics(topics="all"):
    if topics != "all":
        for topic in topics:
            assert topic in camel_topic_risk_ratings
    # dataset = load_dataset("camel-ai/biology", split="train")  # this is really slow
    path = hf_hub_download(
        repo_id="camel-ai/biology",
        repo_type="dataset",
        filename="biology.zip",
        # local_dir="datasets/",
        # local_dir_use_symlinks=False,
    )

    # Get the path to the downloaded file
    zip_path = Path(path)
    # Create an extraction directory in the same parent folder
    extract_dir = zip_path.parent / "extracted"
    # Delete if exists
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(exist_ok=True)
    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find the extracted data file(s)
    # Assuming there's a JSON file with the data
    l_str = [str(path) for path in extract_dir.glob("**/*.json")]
    # Create a dataset from the extracted data
    dataset = load_dataset("json", data_files=l_str, split="train")
    if topics == "all":
        return dataset
    else:
        return dataset.filter(lambda x: x["topic;"] in topics)


# # note that it has Biochemistry too
# def _load_camel_chemistry():
#     # if topics != "all":
#     #     for topic in topics:
#     #         assert topic in camel_topic_risk_ratings
#     # dataset = load_dataset("camel-ai/biology", split="train")  # this is really slow
#     path = hf_hub_download(
#         repo_id="camel-ai/chemistry",
#         repo_type="dataset",
#         filename="chemistry.zip",
#         # local_dir="datasets/",
#         # local_dir_use_symlinks=False,
#     )

#     # Get the path to the downloaded file
#     zip_path = Path(path)
#     # Create an extraction directory in the same parent folder
#     extract_dir = zip_path.parent / "extracted"
#     # Delete if exists
#     if extract_dir.exists():
#         shutil.rmtree(extract_dir)
#     extract_dir.mkdir(exist_ok=True)
#     # Extract the zip file
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extract_dir)

#     # Find the extracted data file(s)
#     # Assuming there's a JSON file with the data
#     l_str = [str(path) for path in extract_dir.glob("**/*.json")]
#     # Create a dataset from the extracted data
#     dataset = load_dataset("json", data_files=l_str, split="train")
#     # if topics == "all":
#     #     return dataset
#     # else:
#     #     return dataset.filter(lambda x: x["topic;"] in topics)
#     return dataset


dataset_loaders = dict(
    wikitext=load_wikitext,
    python=load_python_dataset,
    oscar_en=lambda tokenizer: load_one_oscar_shard("en", tokenizer),
    oscar_pl=lambda tokenizer: load_one_oscar_shard("pl", tokenizer),
    # oscar_es=lambda tokenizer: load_one_oscar_shard("es", tokenizer),
    cruelty=load_cruelty,
    beaver_safe=load_beaver_safe,
    pile_bio_forget=load_pile_bio_forget,
    pile_bio_retain=load_pile_bio_retain,
)


class CachedBatches:
    def __init__(self, base_iter, batch_size):
        assert isinstance(base_iter, IterableDataset)
        self.base_iter = looping_iter(base_iter)
        self.batch_size = batch_size
        self.cache = []

    def __iter__(self):
        yield from self.cache
        while True:
            new_item = get_batch(self.base_iter, self.batch_size)
            self.cache.append(new_item)
            yield new_item


# other datasets
# https://huggingface.co/datasets/lapisrocks/pile-bio
# https://huggingface.co/datasets/lapisrocks/camel-bio
#     https://huggingface.co/datasets/camel-ai/biology
#     this is the original, uncut - better to use it
# https://huggingface.co/datasets/lapisrocks/magpie-bio-filtered
#     it's not really bio, it's just instruction retain set
