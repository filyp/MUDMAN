# %%
import logging
from datetime import datetime
from types import SimpleNamespace

import optuna
import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loading import CachedBatches, dataset_loaders
from utils.git_and_reproducibility import *
from utils.model_operations import *
from utils.training import MockTrial, loss_fns, set_seeds

config = SimpleNamespace(
    # Model/data configs
    model_id="EleutherAI/pythia-14m",
    forget_set_name="python",
    # Training constants
    unlearn_steps=100,
    batch_size=16,
    # Relearning params
    relearn_steps=50,
    relearn_lr=3e-4,
    eval_batch_size=16,
    relearn_lora_conf=dict(r=1, target_modules="all-linear", lora_dropout=0.05),
    # Default tunable params
    disruption_score_warmup=10,
)

pt.set_default_device("cuda")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

# load datasets
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
retain_set = dataset_loaders["wikitext"](tokenizer)
forget_set = dataset_loaders[config.forget_set_name](tokenizer)
retain_batches = CachedBatches(retain_set["train"], config.batch_size)
forget_batches = CachedBatches(forget_set["train"], config.batch_size)
retain_val_batches = CachedBatches(retain_set["validation"], config.eval_batch_size)
forget_val_batches = CachedBatches(forget_set["validation"], config.eval_batch_size)
r_eval_batch = next(retain_val_batches.fresh_iterator())
f_eval_batch = next(forget_val_batches.fresh_iterator())

base_model = AutoModelForCausalLM.from_pretrained(config.model_id)
init_forget = eval_loss(base_model, f_eval_batch)
init_retain = eval_loss(base_model, r_eval_batch)
logging.info(f"init forget: {init_forget:6.2f}    init retain: {init_retain:6.2f}")

_circuit_name = f"{config.forget_set_name}_correct_logit.pt"
circuit = pt.load(repo_root() / "circuits" / config.model_id / _circuit_name)


# %%
def objective(trial):
    # ! parameters
    quantile = trial.suggest_float("quantile", 0.0003, 0.003, log=True)
    unlearn_lr = trial.suggest_float("unlearn_lr", 0.0003, 0.003, log=True)
    retain_amp = trial.suggest_float("retain_amp", 1, 1.4)
    forget_amp = trial.suggest_float("forget_amp", 1, 1.2)
    disruption_score_decay = trial.suggest_float("disruption_score_decay", 0.0, 0.95)

    # prepare data iterators
    retain_iter = retain_batches.fresh_iterator()
    # load model (copy from memory for speed)
    model = deepcopy(base_model)

    target_modules = ["dense_4h_to_h", "dense"]  # for python keep these two
    interven_params = []
    for name, param in model.named_parameters():
        if any(f"{m}.weight" in name for m in target_modules):
            interven_params.append(param)
            # initialize disruption scores
            param.disruption_score = pt.zeros_like(param)
            # initialize to_forget
            param.to_forget = circuit[name]

    # initialize optimizers
    optimizer = pt.optim.SGD(interven_params, lr=unlearn_lr)
    # initialize mask
    mask_fn = lambda param: param.disruption_score / param.to_forget.abs() ** forget_amp

    # ! unlearning loop
    res = {}
    logging.info("step      base_f      base_r")
    for step in range(1, 1 + config.unlearn_steps):
        model.train()
        r_input_ids = next(retain_iter)

        # ! update disruption scores
        model.zero_grad(set_to_none=True)
        loss = loss_fns["cross_entropy"](model(r_input_ids), r_input_ids)
        loss.backward()
        for param in interven_params:
            param.disruption_score *= disruption_score_decay
            param.disruption_score += param.grad.abs() ** retain_amp
        if step <= config.disruption_score_warmup:
            continue

        # ! unlearn on the base model
        # get threshold
        final_scores = [mask_fn(p) for p in interven_params]
        threshold = get_threshold(quantile, final_scores)
        # apply mask
        model.zero_grad(set_to_none=True)
        for param in interven_params:
            mask = mask_fn(param) < threshold
            param.grad = mask * param.to_forget
        optimizer.step()

        # ! eval
        if step % 10 == 0:
            res = dict(
                base_forget=eval_loss(model, f_eval_batch),
                base_retain=eval_loss(model, r_eval_batch),
            )
            logging.info(f"{step:4} " + " ".join(f"{v:11.2f}" for v in res.values()))

            # prune if nan
            if any(pt.isnan(v) for v in res.values()):
                logging.error("NaN in eval results")
                raise optuna.TrialPruned()
            if res["base_retain"] > init_retain + 0.1:
                logging.info("Retain performance broken")
                raise optuna.TrialPruned()

    # ! final bigger eval relearning
    copied_model = deepcopy(model)
    retain_val_iter = retain_val_batches.fresh_iterator()
    forget_val_iter = forget_val_batches.fresh_iterator()
    forget_loss = relearn(copied_model, config, retain_val_iter, forget_val_iter)
    return forget_loss


# %%
info = f"S&D,{config.forget_set_name},{config.relearn_steps}rs"
study_name = f"{info},no_retain_lora,better_ranges"
if __name__ == "__main__":
    assert is_repo_clean()
    study = optuna.create_study(
        study_name=study_name,
        storage=get_storage(),
        direction="maximize",
        # load_if_exists=True,
    )
    save_script_and_attach_logger(__file__, study.study_name)
    study.set_metric_names(["forget_loss"])
    study.set_user_attr("commit_hash", commit_hash())
    for k, v in config.__dict__.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=10000)
