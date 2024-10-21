# %%
from copy import deepcopy
import sys
from pathlib import Path

import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the main directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import device, forward, load_one_oscar_shard

from fading_backprop import (
    get_norm_of_weights_change,
    install_hooks_for_fading_backprop,
    install_hooks_for_saving_gradients,
    scale_perturbation,
    set_fade_factor,
    normal_train_step,
)

# model_id = "google/gemma-2-2b"
model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pl_dataset = load_one_oscar_shard("pl", tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
model.to(device)

install_hooks_for_saving_gradients(model)
install_hooks_for_fading_backprop(model)

# freeze all but mlp down_proj
for param in model.parameters():
    param.requires_grad = False
for layer in model.model.layers:
    layer.mlp.down_proj.weight.requires_grad = True

# %%
original_state_dict = deepcopy(model.state_dict())

# %% test that partial_norm calculation is valid
l = pt.Tensor([1, 2, 3, 4, 5, 6])
assert l.norm() == pt.Tensor([l[:3].norm(), l[3:].norm()]).norm()

# %% at the beginning, the norm of the weights change should be 0
assert 0 == get_norm_of_weights_change(model, original_state_dict)

# %%
batch = next(iter(pl_dataset["unlearn"].batch(1)))
normal_train_step(model, batch, 0.0001)
initial_norm = get_norm_of_weights_change(model, original_state_dict)
assert initial_norm > 0
initial_norm

# %% scaling by 1 should not change the norm
scale_perturbation(model, original_state_dict, 1)
norm = get_norm_of_weights_change(model, original_state_dict)
assert norm == initial_norm

# %% scaling by 0.5 should halve the norm
scale_perturbation(model, original_state_dict, 0.5)
norm = get_norm_of_weights_change(model, original_state_dict)
assert (norm - initial_norm * 0.5).abs() < 1e-4