# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # necessary for determinism:

import logging
from types import SimpleNamespace
from copy import deepcopy

import hydra
import torch as pt
from omegaconf import ListConfig, OmegaConf

from unlearning import unlearn
from utils.data_loading import load_batches
from utils.evals import eval_on
from utils.git_and_reproducibility import *

# from utils.model_operations import relearn, relearn_with_retain
from utils.training import set_seeds

pt.set_default_device("cuda")

s_cfg = SimpleNamespace(
    model_id="meta-llama/Llama-3.2-1B",
    forget_set_name="wmdp_deduped_unlearning",
    retain_set_name="mmlu_retain",
    unlearning_epochs=1,
    batch_size=4,
    target_modules=["up_proj", "down_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"],  # fmt: skip
)
# construct hyperparams
hyperparams = SimpleNamespace(
    normalize_grads=True,
    unlearning_loss_fn="neg_entropy",
    use_masking=True,
    train_adversary=True,
    adv_decay=1,
    adv_lr=0.0003,
    fork_every_n_loops=30,
    retain_momentum=0.8,
    retaining_rate=1.0e-7,
    unlearning_rate=1.0e-10,
    # square_norm=False,
)

print("loading datasets")
# load datasets
set_seeds(42)
forget_batches = load_batches(
    s_cfg.model_id,
    s_cfg.forget_set_name,
    s_cfg.batch_size,
)
retain_batches = load_batches(
    s_cfg.model_id,
    s_cfg.retain_set_name,
    s_cfg.batch_size,
)

# %%
import torch as pt
from transformers import AutoModelForCausalLM
from utils.loss_fns import cross_entropy_loss

model = AutoModelForCausalLM.from_pretrained(
    s_cfg.model_id, torch_dtype=pt.bfloat16
)

# %%
def correct_logit_minus_avg_loss(output, batch, clip_at=0):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
    ids = batch["input_ids"][:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    true_logits -= logits.mean(dim=-1)
    true_logits = true_logits.clip(min=clip_at)

    attn_mask = batch["attention_mask"][:, 1:].flatten()
    true_logits *= attn_mask
    return true_logits.sum() / attn_mask.sum()



model.zero_grad(set_to_none=True)
batch = forget_batches[14]
out = model(**batch, output_hidden_states=True)

batch = deepcopy(batch)
# batch["attention_mask"] *= 0
# batch["attention_mask"] = pt.ones_like(batch["input_ids"])

retain_loss = correct_logit_minus_avg_loss(out, batch)
retain_loss


# %%

set_seeds(42)
model = unlearn(
    hyperparams,
    s_cfg,
    retain_batches,
    forget_batches,
)
