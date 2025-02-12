# %%
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import optuna
import yaml

# add to python path __file__.parent.parent
sys.path.append(str(Path(__file__).parent.parent))
from utils.git_and_reproducibility import *
from utils.git_and_reproducibility import repo_root
from utils.model_operations import *
from utils.plots_and_stats import *
from utils.training import get_stats_from_last_n_trials

plt.style.use("default")  # Reset to default style

db_url = json.load(open(repo_root() / "secret.json"))["db_url"]
storage = get_storage(db_url)

# %% get the studies
config_path = repo_root() / "configs" / "wmdp3.yaml"
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

multistudy_name = Path(config_path).stem

config = SimpleNamespace(**full_config["general_config"])
relearn_config = SimpleNamespace(**full_config["relearn_config"])

method_stats = dict()
for variant_name in full_config["variants"]:
    study_name = (
        f"{config.unlearn_steps},{relearn_config.relearn_steps},"
        f"{config.forget_set_name}"
        f"|{multistudy_name}|{variant_name}"
    )
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Get stats for the last N trials instead of just best trial
    trials = study.get_trials()
    markdown_line, last_n_mean, last_n_sem = get_stats_from_last_n_trials(
        study, trials, n=20
    )
    method_stats[variant_name] = (last_n_mean, last_n_sem)
    # method_stats[variant_name] = study.best_trial.value

method_stats

# %%
titles_dict = {
    "TAR2": "TAR",
    # "neg_cross_entropy_loss": "MUDMAN",
    "neg_entropy_loss": "MUDMAN",
    "no_adversary_ent": "w/o meta-learning",
    "no_masking_ent": "w/o masking",
    "no_normalization_ent": "w/o normalization",
    # "logit_loss": "logit loss",
    # "no_r_momentum": "no retain momentum",
    # "no_adv_decay": "no adversary decay",
}
positions_dict = {
    "TAR2": 4,
    "neg_entropy_loss": 3,
    "no_adversary_ent": 2,
    "no_masking_ent": 1,
    "no_normalization_ent": 0,
    # "neg_entropy_loss": 0,
    # "logit_loss": 6,
    # "no_r_momentum": 3,
    # "no_adv_decay": 1,
}

# Create a single plot
fig, ax = plt.subplots(figsize=(4.4, 2.0))

# Set title and labels
ax.set_title("Llama-3.2-1B unlearned on Pile-Bio", fontsize=12, x=0.73)
ax.set_xlabel("WMDP-Bio accuracy after relearning↓")

# Create a color mapping for methods
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_method_names = list(titles_dict.keys())
method_to_color = {name: color for name, color in zip(_method_names, colors)}

# Get baseline value
baseline_path = repo_root() / "results" / "baselines" / f"{config_path.stem}.txt"
baseline = float(baseline_path.read_text())

# Calculate the differences from baseline
differences = [mean - baseline for mean, sem in method_stats.values()]
sems = [sem for mean, sem in method_stats.values()]

ax.barh(
    [positions_dict[name] for name in method_stats.keys()],
    differences,
    xerr=sems,
    height=1,
    capsize=3,
    color=[method_to_color[name] for name in method_stats.keys()],
    left=baseline  # Start bars from baseline
)

# Update yticks
ax.set_yticks([positions_dict[name] for name in method_stats.keys()])
ax.set_yticklabels([titles_dict[name] for name in method_stats.keys()])

# Remove spines
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)

# Set xlim
# Set xlim (keeping same range but inverted from baseline)
ax.set_xlim(0.36, baseline)
ax.yaxis.tick_right()

# ax.yaxis.set_label_position("right")

plt.tight_layout()

# %%

# %%

plot_path = repo_root() / "paper" / "plots" / "wmdp.pdf"
fig.savefig(plot_path)
