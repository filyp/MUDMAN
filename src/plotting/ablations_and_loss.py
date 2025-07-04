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
multistudy_names = [
    # "llama32,python",
    # "llama32,pile-bio",
    # "smol,python",
    # "smol,pile-bio",
    # "pythia,python",
    # "pythia,pile-bio",
    "llama32,pile-bio",
    "smol,pile-bio",
    "pythia,pile-bio",
    "llama32,python",
    "smol,python",
    "pythia,python",
]
for multistudy_name in multistudy_names:
    multistudy_to_method_stats[multistudy_name] = dict()

    # load YAML configuration
    config_path = (
        repo_root() / "configs" / f"ablations_and_loss2,{multistudy_name}.yaml"
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
            # print(f"Study {study_name} not found")
            continue

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            # print(f"Study {study_name} has no complete trials!")
            continue

        # get stats for the last N trials
        trials = study.get_trials()
        markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(
            study, trials, n=30
        )
        multistudy_to_method_stats[multistudy_name][variant_name] = (
            last_n_mean,
            last_n_sem,
            # study.best_trial.values[0],
            # 0,
        )

        # # check if optimal values are near range edges
        # make_sure_optimal_values_are_not_near_range_edges(study)


# %%
titles_dict = {
    "TAR2": "TAR",
    "neg_cross_entropy_loss": "MUDMAN",
    "no_adversary": "w/o meta-learning",
    "no_masking": "w/o masking",
    "no_normalization": "w/o normalization",
    # "neg_entropy_loss": "MUDMAN w/ neg entropy loss",
    # "logit_loss": "logit loss",
    # "no_r_momentum": "no retain momentum",
    # "no_adv_decay": "no adversary decay",
}
positions_dict = {
    "TAR2": 4,
    "neg_cross_entropy_loss": 3,
    "no_adversary": 2,
    "no_masking": 1,
    "no_normalization": 0,
    # "neg_entropy_loss": 0,
    # "logit_loss": 6,
    # "no_r_momentum": 3,
    # "no_adv_decay": 1,
}

# Create the plot with n subplots side by side
fig, axes = plt.subplots(2, 3, figsize=(9, 3.3))
# todo post-review: make the plot higher, to relax a bit; (maybe also add the safeguarding loss? nah)

# Set column titles with specified font size
column_fontsize = 12  # Adjust this value as needed
axes[0, 0].set_title("Llama-3.2-1B", fontsize=column_fontsize)
axes[0, 1].set_title("SmolLM-135M", fontsize=column_fontsize)
axes[0, 2].set_title("pythia-14m", fontsize=column_fontsize)
axes[-1, 0].set_xlabel("Forget loss after relearning↑")
axes[-1, 1].set_xlabel("Forget loss after relearning↑")
axes[-1, 2].set_xlabel("Forget loss after relearning↑")

# Set row titles with the same font size
axes[0, 0].set_ylabel("Pile-Bio", fontsize=column_fontsize, labelpad=12)
axes[1, 0].set_ylabel("Python", fontsize=column_fontsize, labelpad=12)

# Create a color mapping for methods
# Use default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_method_names = list(titles_dict.keys())
method_to_color = {name: color for name, color in zip(_method_names, colors)}
method_to_color

# Define data for plotting
data = []  # Initialize data list
scales = [0.5, 6]

for n, (multistudy_name, method_stats) in enumerate(multistudy_to_method_stats.items()):
    ax = axes[n // 3, n % 3]

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

    # Update yticks for reversed order
    ax.set_yticks([positions_dict[name] for name in method_stats.keys()])
    if n % 3 == 0:
        # ax.yaxis.set_tick_params(pad=150)  # Adjust this value as needed
        ax.set_yticklabels([titles_dict[name] for name in method_stats.keys()])
    else:
        ax.set_yticklabels([])

    # ax.set_title(multistudy_name)  # todo when reoadring plots, reenable this to see if manual labels are still valid

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    config_path = (
        repo_root() / "configs" / f"ablations_and_loss2,{multistudy_name}.yaml"
    )
    baseline_path = repo_root() / "results" / "baselines" / f"{config_path.stem}.txt"
    if baseline_path.exists():
        baseline = float(baseline_path.read_text())
        # Add baseline
        ax.axvline(x=baseline, color="black", linestyle="--", alpha=0.3)

    # Calculate the minimum and maximum mean values for the xlim
    max_bar = max(mean for mean, sem in method_stats.values())
    min_bar = min(mean for mean, sem in method_stats.values())
    if baseline_path.exists():
        min_bar = min(min_bar, baseline)
    # margin = (max_bar - min_bar) / 5  # Calculate the margin
    center = (max_bar + min_bar) / 2
    scale = scales[n // 3]
    ax.set_xlim(center - scale / 2, center + scale / 2)

plt.tight_layout()


plot_path = repo_root() / "paper" / "latex" / "plots" / "ablations_and_loss.pdf"
if plot_path.parent.exists():
    print(f"Saving plot to {plot_path}")
    fig.savefig(plot_path)
else:
    plt.show()  # Ensure the plot is displayed
