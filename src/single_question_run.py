# %%
# %load_ext autoreload
# %autoreload 2
import logging
from copy import deepcopy

import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

import wandb
from datasets import load_dataset
from utils import loss_fns
from utils.data_loading import (
    data_paths,
    filter_by_question,
    load_batches,
    load_low_mi_set,
    load_retain_corpus,
)
from utils.evals import eval_on
from utils.loss_fns import cross_entropy_loss, loss_fns
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")

s = OmegaConf.create(
    dict(
        model_id="meta-llama/Llama-3.2-1B",
        unlearning_epochs=12,
        max_length=128,
    )
)

# %%
# load corpora
f_full_corpus = load_low_mi_set(data_paths["wmdp_deduped_unlearning"])
r_full_corpus = load_low_mi_set(data_paths["wmdp_deduped_wrong_answers"])

# load questions
wmdp_mcq_full = load_low_mi_set(data_paths["wmdp_deduped_mcq_full"])

# load disrution eval set
d_corpus = load_retain_corpus("fineweb_edu")
disruption_batches = load_batches(d_corpus, s.model_id, 16, s.max_length)

# # mmlu_set = load_dataset("cais/mmlu", "college_biology", split="test")
# mmlu_set = load_dataset("cais/mmlu", "all", split="validation")
# mmlu_set = mmlu_set.shuffle(seed=0).select(range(64))

# %%
# choose eval question
question_index = 3
f_eval_set = wmdp_mcq_full.select([question_index])
target_question = f_eval_set[0]["question"]
print(f"{target_question=}")

# load forget set
f_corpus = f_full_corpus.filter(lambda ex: ex["original_question"] == target_question)
forget_batches = load_batches(f_corpus, s.model_id, 3, s.max_length)
assert len(forget_batches) == 1

# load retain set
r_corpus = r_full_corpus.filter(lambda ex: ex["original_question"] == target_question)
retain_batches = load_batches(r_corpus, s.model_id, 3, s.max_length)
assert len(retain_batches) == 1

# wmdp_acc = eval_on(f_eval_set, model, temperature=1)
# %%
def unlearn_percentiles(
    h,
    conf,
    retain_batches,
    forget_batches,
    eval_callback,
):
    loss_fn = loss_fns[h.unlearning_loss_fn]

    set_seeds(42)
    model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.bfloat16)
    model.config.use_cache = False

    # ! retain pass
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(**retain_batches[0])
    retain_loss = cross_entropy_loss(output, retain_batches[0])
    retain_loss.backward()
    for p in model.parameters():
        p.retain_acc = p.grad

    # ! forget pass
    model.zero_grad(set_to_none=True)
    output = model(**forget_batches[0])
    forget_loss = loss_fn(output, forget_batches[0], 0)
    forget_loss.backward()

    grad_norm = 0
    for n, p in model.named_parameters():
        if h.modules not in n:
            continue

        # ! limit percentiles
        if h.percentile is not None:
            abs_vals = p.grad.flatten().abs()
            k = int(len(abs_vals) * h.percentile)
            cutoff = abs_vals.kthvalue(k).values.item()
            mask = p.grad.abs() > cutoff
            p.grad *= mask

        # ! masking
        if h.use_masking:
            mask = p.retain_acc.sign() == p.grad.sign()
            p.grad *= mask
        
        grad_norm += p.grad.norm() ** 2
    grad_norm = grad_norm ** 0.5

    # ! unlearning loop
    update_norm = 0
    for loop_num in range(conf.unlearning_epochs):
        eval_callback(model, update_norm)

        for n, p in model.named_parameters():
            if h.modules not in n:
                continue
            # ! update weights
            p.data -= h.unlearning_rate * p.grad

        update_norm += h.unlearning_rate * grad_norm

    return model


def _eval_callback(model, update_norm):
    model.eval()
    # eval mmlu and wmdp
    with pt.no_grad():
        wmdp_acc = eval_on(f_eval_set, model, temperature=1)
        # mmlu_acc = eval_on(mmlu_set, model, temperature=1)

        loss = 0
        for d_batch in disruption_batches[-32:]:
            output = model(**d_batch)
            loss += cross_entropy_loss(output, d_batch)
        disr_loss = loss / len(disruption_batches[-32:])

    update_norm2 = update_norm ** 2
    logging.info(f"{wmdp_acc=:.4f} {disr_loss=:.4f} {update_norm2=:.4f}")
    wandb.log({"wmdp_acc": wmdp_acc, "disr_loss": disr_loss, "update_norm2": update_norm2})
    if wmdp_acc < 0.3 or disr_loss > 2.62:
        raise StopIteration


_lr = 3e-3
for modules, unlearning_rate in [
    ("up_proj", _lr),
    ("down_proj", _lr * 0.1),
    ("gate_proj", _lr * 4),
    ("q_proj", _lr),
    ("k_proj", _lr * 4),
    ("v_proj", _lr),
    ("o_proj", _lr),
]:
    # construct hyperparams
    h = OmegaConf.create(
        dict(
            normalize_grads=False,
            unlearning_loss_fn="neg_cross_entropy",
            # unlearning_loss_fn="neg_entropy",
            # unlearning_loss_fn="correct_logit_minus_avg",
            #
            use_masking=False,
            unlearning_rate=unlearning_rate,
            modules=modules,
            percentile=None,
        )
    )
    s.unlearning_epochs = 30
    name = f"{h.unlearning_loss_fn}|{h.unlearning_rate}|{h.modules}"
    wandb.init(project="wmdp_single2", name=name, group="modules_exp")
    try:
        unlearn_percentiles(
            h,
            s,
            retain_batches,
            # disruption_batches,
            forget_batches,
            _eval_callback,
        )
    except StopIteration:
        logging.info("Stopping early")
    pt.cuda.empty_cache()
    wandb.finish()


# %%
