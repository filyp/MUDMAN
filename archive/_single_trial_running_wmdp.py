# %%
from IPython import get_ipython


# automatically reload all modules
ipython = get_ipython()
if ipython is not None:  # Only runs in IPython environment
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
# necessary for determinism:
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

import logging
from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch as pt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.loss_fns import loss_fns
from utils.model_operations import relearn, relearn_with_retain
from utils.training import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# style default
plt.style.use("default")

# %%
# load YAML configuration
config_path = repo_root() / "configs" / "wmdp7.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

config = full_config["general_config"]

config = SimpleNamespace(**config)
relearn_config = SimpleNamespace(**full_config["relearn_config"])

# %%

pt.set_default_device("cuda")

# load datasets
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders[config.retain_set_name](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.batch_size)
r_eval = next(iter(retain_val_batches))
f_eval = next(iter(forget_val_batches))

_init_res = eval_(
    AutoModelForCausalLM.from_pretrained(config.model_id), f_eval, r_eval
)
allowed_r_loss = _init_res["retain_loss"]

# %%

hyperparams = SimpleNamespace(
    additional_param=None,
    adv_decay=1,
    adv_lr=0.001,
    fork_every_n_loops=48,
    retain_momentum=0.95,
    retaining_rate=0.001,
    # unlearning_rate=10e-6,
    unlearning_rate=1e-6,
)
config.unlearning_loss_fn = "neg_entropy"
config.use_masking = False
# config.use_masking = True

# note, not training adverary results in higher base_r loss
pt.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    config.model_id, torch_dtype=pt.bfloat16
)

set_seeds(42)
model, _results = surgical_irreversible_unlearning(
    hyperparams,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,

    # allowed_r_loss=_init_res["retain_loss"] + config.hard_loss_budget,
    allowed_r_loss=float("inf"),
    model=model,
    # soft_threshold=_init_res["retain_loss"] + config.soft_loss_budget,
    # eval_wmdp_every=config.eval_wmdp_every,
    eval_wmdp_every=9999999,
    allowed_mmlu_acc=config.allowed_mmlu_acc,
)

# %%
# plot results

import matplotlib.pyplot as plt

plt.style.use("default")  # Reset to default style

x = [r["step"] for r in _results]
y1 = [r["retain_loss"].item() for r in _results]
y2 = [r["forget_loss"].item() for r in _results]

fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.plot(x, y1, label="retain loss")
ax.plot(x, y2, label="forget loss")
ax.legend()
ax.set_xlabel("Step")

# Add tight layout with padding to prevent clipping
plt.tight_layout(pad=1.1)  # Adjust the padding value as needed

# Save the figure before showing it
plot_path = repo_root() / "paper" / "latex" / "plots" / "logo" / "no_masking.pdf"
# plot_path = repo_root() / "paper" / "latex" / "plots" / "logo" / "masking.pdf"
plt.savefig(plot_path, bbox_inches='tight')  # Use bbox_inches='tight' to include all elements

# Now show it (this will clear the figure)
plt.show()

# note that the one with masking uses 10x the unlearning rate



