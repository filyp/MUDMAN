# %%
# to run all variants one after another:
# python study_runner.py PATH_TO_CONFIG [if_study_exists=fail|delete|load]

# necessary for determinism:
from copy import deepcopy
import os

import hydra
from hydra.core.hydra_config import HydraConfig

from utils.data_loading_lowMI import data_paths, load_batches
from utils.evals import eval_on

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"  # less mem but slower

from types import SimpleNamespace

import torch as pt
from omegaconf import ListConfig, OmegaConf

from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from utils.git_and_reproducibility import *
from utils.model_operations import relearn, relearn_with_retain
from utils.training import set_seeds


@hydra.main(version_base="1.2", config_path="../configs", config_name="wmdp_main")
def run_study(full_config):
    variant_num = full_config.variant_num
    print(f"{variant_num=}")
    study_config = full_config.study_config

    storage = get_storage()
    pt.set_default_device("cuda")

    # split into hyperparams and other keys
    variant_name, custom = list(full_config.variants.items())[variant_num]
    hyperparam_ranges = OmegaConf.merge(full_config.hyperparams, custom)

    study_name = f"test|{variant_name}"
    print(f"{study_name=}")
    print(f"{hyperparam_ranges=}")

    # load datasets
    set_seeds(42)
    # todo configure
    # forget_batches = load_batches(config.model_id, data_paths["wmdp_deduped_unlearning"])
    forget_batches = load_batches(
        study_config.model_id, data_paths["wmdp_deduped_unlearning"][-1:]
    )
    retain_batches = load_batches(study_config.model_id, data_paths["mmlu_retain"])

    def objective(trial):
        # construct hyperparams
        hyperparams = deepcopy(hyperparam_ranges)
        for hp_name, distribution in hyperparams.items():
            if isinstance(distribution, ListConfig):
                low, high, log = distribution
                hyperparams[hp_name] = trial.suggest_float(hp_name, low, high, log=log)
        logging.info(f"trial {trial.number} - {trial.params}")

        set_seeds(42)
        model = surgical_irreversible_unlearning(
            hyperparams,
            study_config,
            retain_batches,
            forget_batches,
        )

        # set_seeds(42)
        # todo relearning
        # forget_losses = relearn(
        #     model, relearn_config, retain_val_batches, forget_val_batches
        # )

        wmdp_accuracy = eval_on("wmdp_deduped_mcq_eval", model)
        print(f"{wmdp_accuracy=}")

        # use min rather than last, in case it anomalously increases
        return wmdp_accuracy

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.set_metric_names(["wmdp_accuracy"])
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())

    print(hydra.job)
    # Add Hydra run information
    # hydra_config = HydraConfig.get()
    # study.set_user_attr("hydra_run_dir", str(hydra_config.run.dir))
    # study.set_user_attr("hydra_job_num", hydra_config.job.num)
    # study.set_user_attr("hydra_job_id", hydra_config.job.id)

    # run remaining trials
    n_trials = max(0, study_config.n_trials - len(study.trials))
    print(f"{n_trials=}")
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    run_study()
