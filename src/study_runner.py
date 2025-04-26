# %%
# # necessary for determinism:
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hashlib
import os
from copy import deepcopy

import hydra
import torch as pt
from omegaconf import ListConfig, OmegaConf

from unlearning_methods.surgical_irreversible_unlearning import (
    surgical_irreversible_unlearning,
)
from utils.data_loading_lowMI import load_batches
from utils.evals import eval_on
from utils.git_and_reproducibility import *
from utils.model_operations import relearn, relearn_with_retain
from utils.training import set_seeds


@hydra.main(version_base="1.2", config_path="../configs", config_name="wmdp_main")
def run_study(cfg):
    study_config = cfg.study_config

    storage = get_storage()
    pt.set_default_device("cuda")

    # split into hyperparams and other keys
    variant_name, custom = list(cfg.variants.items())[cfg.variant_num]
    hyperparam_ranges = OmegaConf.merge(cfg.hyperparams, custom)

    config_hash = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
    study_name = f"{config_hash[:4]}|{variant_name}"
    print(f"{study_name=}")
    print(f"{hyperparam_ranges=}")

    # load datasets
    set_seeds(42)
    # todo configure
    forget_batches = load_batches(study_config.model_id, "wmdp_deduped_unlearning")
    retain_batches = load_batches(study_config.model_id, "mmlu_retain")

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
        mmlu_accuracy = eval_on("filtered_mmlu", model)
        print(f"{wmdp_accuracy=} {mmlu_accuracy=}")
        # use min rather than last, in case it anomalously increases
        return wmdp_accuracy, mmlu_accuracy

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=False,
    )
    study.set_metric_names(["wmdp_accuracy", "mmlu_accuracy"])
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())

    # # run remaining trials
    # n_trials = max(0, study_config.n_trials - len(study.trials))
    # print(f"{n_trials=}")
    # study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    run_study()


# todo train only on attention_mask=1