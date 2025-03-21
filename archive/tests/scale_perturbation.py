# %%
import sys
from copy import deepcopy
from pathlib import Path

# Add the main directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from common_startup_code import *
from fading_backprop import install_hooks_for_saving_gradients
from utils import get_norm_of_weights_change, normal_train_step, scale_perturbation

install_hooks_for_saving_gradients(model)

# freeze all but mlp down_proj
for param in model.parameters():
    param.requires_grad = False
for layer in model.model.layers:
    layer.mlp.down_proj.weight.requires_grad = True

# %%
original_state_dict = deepcopy(model.state_dict())

# %% test that partial_norm calculation is valid
l = pt.tensor([1, 2, 3, 4, 5, 6], dtype=pt.float32)
assert l.norm() == pt.tensor([l[:3].norm(), l[3:].norm()]).norm()

# %% at the beginning, the norm of the weights change should be 0
assert 0 == get_norm_of_weights_change(model, original_state_dict)

# %%
batch = next(iter(forget_set["unlearn"].batch(1)))
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
