# %%
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml

# add to python path __file__.parent.parent
sys.path.append(str(Path(__file__).parent.parent))
from utils.git_and_reproducibility import *
from utils.git_and_reproducibility import repo_root
from utils.model_operations import *
from plotting.plots_and_stats import *
from utils.training import (
    get_stats_from_last_n_trials,
    make_sure_optimal_values_are_not_near_range_edges,
)

plt.style.use("default")  # Reset to default style

storage = get_storage()

# %% get the studies
multistudy_to_method_stats = dict()
# Just pick one multistudy
multistudy_name = "smol,python"
multistudy_to_method_stats[multistudy_name] = dict()

# load YAML configuration
config_path = (
    repo_root() / "configs" / f"ablations_and_loss3,{multistudy_name}.yaml"
)
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])

for variant_name in full_config["variants"]:
    study_name = (
        f"{config.unlearn_steps},{relearn_config.relearn_steps},"
        f"{config.forget_set_name}"
        f"|{Path(config_path).stem}|{variant_name}"
    )

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        continue

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        continue

    # get stats for the last N trials
    trials = study.get_trials()
    markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(
        study, trials, n=30
    )
    multistudy_to_method_stats[multistudy_name][variant_name] = (
        last_n_mean,
        last_n_sem,
    )

# %%
titles_dict = {
    "TAR2": "TAR",
    "neg_cross_entropy_loss": "MUDMAN",
    "no_adversary": "w/o meta-learning",
    "no_masking": "w/o masking",
    "no_normalization": "w/o normalization",
}
positions_dict = {
    "TAR2": 4,
    "neg_cross_entropy_loss": 3,
    "no_adversary": 2,
    "no_masking": 1,
    "no_normalization": 0,
}

# Create a single plot
fig, ax = plt.subplots(figsize=(4, 3))

# Set title and labels
# ax.set_title("Pythia-70M on Python", fontsize=12)
ax.set_xlabel("Forget loss after relearning↑")

# Create a color mapping for methods
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_method_names = list(titles_dict.keys())
method_to_color = {name: color for name, color in zip(_method_names, colors)}

method_stats = multistudy_to_method_stats[multistudy_name]
# filter method_stats
method_stats = {n: s for n, s in method_stats.items() if n in titles_dict}

ax.barh(
    [positions_dict[name] for name in method_stats.keys()],
    [mean for mean, sem in method_stats.values()],
    xerr=[sem for mean, sem in method_stats.values()],
    height=1,
    capsize=3,
    color=[method_to_color[name] for name in method_stats.keys()],
)

# Update yticks
ax.set_yticks([positions_dict[name] for name in method_stats.keys()])
ax.set_yticklabels([titles_dict[name] for name in method_stats.keys()])

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

config_path = (
    repo_root() / "configs" / f"ablations_and_loss3,{multistudy_name}.yaml"
)
baseline_path = repo_root() / "results" / "baselines" / f"{config_path.stem}.txt"
if baseline_path.exists():
    baseline = float(baseline_path.read_text())
    # Add baseline
    ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

# # Calculate the minimum and maximum mean values for the xlim
# max_bar = max(mean for mean, sem in method_stats.values())
# min_bar = min(mean for mean, sem in method_stats.values())
# if baseline_path.exists():
#     min_bar = min(min_bar, baseline)
# center = (max_bar + min_bar) / 2
# scale = 0.5  # Using the first scale from the original scales list
# ax.set_xlim(center - scale / 2, center + scale / 2)

plt.tight_layout()

# plot_path = repo_root() / "paper" / "latex" / "plots" / "ablations_and_loss.pdf"
# if plot_path.parent.exists():
#     print(f"Saving plot to {plot_path}")
#     fig.savefig(plot_path)
# else:
#     plt.show()
