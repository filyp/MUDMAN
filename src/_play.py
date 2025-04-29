# %%
%load_ext autoreload
%autoreload 2

import os

from omegaconf import OmegaConf

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # necessary for determinism:

import logging
from copy import deepcopy
from types import SimpleNamespace

import torch as pt
from transformers import AutoModelForCausalLM

from datasets import load_dataset
from unlearning.unlearning import unlearn
from utils.data_loading import (
    data_paths,
    filter_by_question,
    load_batches,
    load_low_mi_set,
    load_retain_corpus,
)
from utils.evals import eval_on
from utils.git_and_reproducibility import *
from utils.training import set_seeds


logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")

s = OmegaConf.create(dict(
    model_id="meta-llama/Llama-3.2-1B",
    forget_set_name="wmdp_deduped_unlearning",
    retain_set_name="fineweb_edu",
    unlearning_epochs=1,
    batch_size=4,
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # fmt: skip
    eval_temperature=0.0,  # Added from study_runner.py
    category="bio",
    portion=0.15,
    max_length=128,
))

# construct hyperparams
hyperparams = OmegaConf.create(dict(
    normalize_grads=True,
    unlearning_loss_fn="neg_entropy",
    use_masking=True,
    train_adversary=True,
    adv_decay=1,
    adv_lr=0.0003,
    fork_every_n_loops=20,
    retain_momentum=0.8,
    retaining_rate=1.0e-7,
    unlearning_rate=1.0e-10,
    clip_at=0,  # Added from unlearning.py
))

# %%
# load forget set
_f_corpus = load_low_mi_set(data_paths[s.forget_set_name])
_f_corpus = filter_by_question(_f_corpus, category=s.category, portion=s.portion)
forget_batches = load_batches(_f_corpus, s.model_id, s.batch_size, s.max_length)

# load retain set
retain_corpus = load_retain_corpus(s.retain_set_name)
retain_batches = load_batches(retain_corpus, s.model_id, s.batch_size, s.max_length)

# # load relearn set
# rel_corpus = load_low_mi_set(data_paths[cfg.relearn_config.set_name])
# rel_corpus = filter_by_question(rel_corpus, category=s.category, portion=s.portion)
# relearn_batches = load_batches(rel_corpus, s.model_id, s.batch_size, s.max_length)

# # load unlearning eval set
# wmdp_set = load_low_mi_set(data_paths["wmdp_deduped_mcq_eval"])
# wmdp_set = filter_by_question(wmdp_set, category=s.category, portion=s.portion)
# mmlu_set = load_dataset("cais/mmlu", "all", split="validation")
# mmlu_set = mmlu_set.shuffle(seed=42).select(range(300))  # todo revert back later

# %%
set_seeds(42)
unlearn(
    hyperparams,
    s,
    retain_batches,
    forget_batches,
    lambda _: None,
)
