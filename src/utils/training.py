import logging
import random

import numpy as np
import optuna
import torch as pt
from transformers import set_seed as set_transformers_seed

import wandb
from utils.loss_fns import cross_entropy_loss


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)
    pt.use_deterministic_algorithms(True)


def make_sure_optimal_values_are_not_near_range_edges(study):
    best_trial = study.best_trial  # ask only once because it's slow
    """Make sure the value is not in the top or bottom 10% of the range."""
    for param_name, param_dist in best_trial.distributions.items():
        min_ = param_dist.low
        max_ = param_dist.high
        value = best_trial.params[param_name]
        if param_dist.log:
            min_ = np.log10(min_)
            max_ = np.log10(max_)
            value = np.log10(value)

        method_name = study.study_name.split("|")[-1]
        if value < min_ + 0.1 * (max_ - min_):
            print(f"\t{param_name}\t in bottom 10% with value {value} in {method_name}")
        if value > max_ - 0.1 * (max_ - min_):
            print(f"\t{param_name}\t in top 10% with value {value} in {method_name}")
            # print(f"WARNING: {param_name} in the top 10% of the range in best trial")
            # print(f"range: {min_} - {max_}, value: {value}, log={param_dist.log}")


# stats for the last n non-pruned trials
def get_stats_from_last_n_trials(study, trials, n=10):
    ok_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"all_trials={len(trials)}, ok_trials={len(ok_trials)}, {study.study_name}")
    values = [t.values[0] for t in ok_trials]

    # max_val = study.best_trial.values[0]
    last_n_mean = np.mean(values[-n:])
    last_n_sem = np.std(values[-n:]) / np.sqrt(n)
    pure_name = study.study_name.split("|")[-1]
    # result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {max_val:.4f} | {pure_name} |  |"
    result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {pure_name} |  |"
    return result, last_n_mean, last_n_sem


def delete_study_if_exists(study_name, storage):
    try:
        _ = optuna.load_study(study_name=study_name, storage=storage)
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass


def relearn(model, config, retain_val_batches, forget_val_batches, use_lora=False):
    for p in model.parameters():
        p.requires_grad = True

    # get batches
    retain_val_iter = iter(retain_val_batches)
    forget_val_iter = iter(forget_val_batches)
    f_eval_batch = next(forget_val_iter)
    r_eval_batch = next(retain_val_iter)

    # if use_lora:
    #     lora_config = LoraConfig(**config.relearn_lora_conf)
    #     peft_model = get_peft_model(model, lora_config, adapter_name="relearning_lora")
    #     model = peft_model.model

    optimizer = pt.optim.SGD(model.parameters(), lr=config.relearn_lr)

    # ! relearning loop
    logging.info("")
    f_losses = []
    # each step is one pass (forward or backward) and a loop has two passes
    passes_per_loop = 2
    for loop_num in range(config.relearn_steps // passes_per_loop):
        # standard forward, backward, and update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        f_input_ids = next(forget_val_iter)
        loss_forget = cross_entropy_loss(model(f_input_ids), f_input_ids)
        loss_forget.backward()
        optimizer.step()

        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 30 == 0:
            res = eval_(model, f_eval_batch, r_eval_batch, step=_passes_done)
            f_losses.append(res["forget_loss"])
            # wandb.log(res, step=_passes_done)

    logging.info("")
    return f_losses


# def only_grad_on(model, params_to_grad):
#     for param in model.parameters():
#         param.requires_grad = False
#     for param in params_to_grad:
#         param.requires_grad = True


# def get_thresh(quantile, disruption_scores):
#     """
#     Calculate threshold value for parameter masking, based on the quantile.
#     For example, if quantile is 0.01, the threshould will cut off 1% of the highest scores.
#     """
#     flat_scores = pt.cat([s.flatten() for s in disruption_scores])
#     return pt.quantile(flat_scores, 1 - quantile, interpolation="lower")


# def copy_model_and_collapse_loras(peft_model, delete_adv=True):
#     """
#     Creates a copy of the model with retention LoRA merged and adversarial LoRA removed.
#     """
#     peft_model_copy = deepcopy(peft_model)
#     # delete adversarial lora
#     if delete_adv:
#         peft_model_copy.delete_adapter("adv_lora")
#     # merge and unload helper lora
#     peft_model_copy.set_adapter(["ret_lora"])
#     collapsed = peft_model_copy.merge_and_unload()
#     del collapsed.peft_config
#     return collapsed

# # --- Mock Trial for Optuna ---
# class MockTrial:
#     def __init__(self, **params):
#         self.params = params
#         self.number = 0
#     def suggest_float(self, name, *args, **kwargs):
#         return self.params[name]
#     def suggest_categorical(self, name, *args, **kwargs):
#         return self.params[name]
#     def suggest_int(self, name, *args, **kwargs):
#         return int(self.params[name])
#     def set_user_attr(self, *args, **kwargs):
#         pass
