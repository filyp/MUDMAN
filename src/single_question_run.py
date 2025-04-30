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
param_names = [
    "embed_tokens.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.1.mlp.up_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.1.input_layernorm.weight",
    "model.layers.1.post_attention_layernorm.weight",
    "model.layers.2.self_attn.q_proj.weight",
    "model.layers.2.self_attn.k_proj.weight",
    "model.layers.2.self_attn.v_proj.weight",
    "model.layers.2.self_attn.o_proj.weight",
    "model.layers.2.mlp.gate_proj.weight",
    "model.layers.2.mlp.up_proj.weight",
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.2.input_layernorm.weight",
    "model.layers.2.post_attention_layernorm.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.3.self_attn.k_proj.weighQt"
    "model.layers.3.self_attn.v_proj.weight",
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.3.mlp.gate_proj.weight",
    "model.layers.3.mlp.up_proj.weight",
    "model.layers.3.mlp.down_proj.weight",
    "model.layers.3.input_layernorm.weight",
    "model.layers.3.post_attention_layernorm.weight",
    "model.layers.4.self_attn.q_proj.weight",
    "model.layers.4.self_attn.k_proj.weight",
    "model.layers.4.self_attn.v_proj.weight",
    "model.layers.4.self_attn.o_proj.weight",
    "model.layers.4.mlp.gate_proj.weight",
    "model.layers.4.mlp.up_proj.weight",
    "model.layers.4.mlp.down_proj.weight",
    "model.layers.4.input_layernorm.weight",
    "model.layers.4.post_attention_layernorm.weight",
    "model.layers.5.self_attn.q_proj.weight",
    "model.layers.5.self_attn.k_proj.weight",
    "model.layers.5.self_attn.v_proj.weight",
    "model.layers.5.self_attn.o_proj.weight",
    "model.layers.5.mlp.gate_proj.weight",
    "model.layers.5.mlp.up_proj.weight",
    "model.layers.5.mlp.down_proj.weight",
    "model.layers.5.input_layernorm.weight",
    "model.layers.5.post_attention_layernorm.weight",
    "model.layers.6.self_attn.q_proj.weight",
    "model.layers.6.self_attn.k_proj.weight",
    "model.layers.6.self_attn.v_proj.weight",
    "model.layers.6.self_attn.o_proj.weight",
    "model.layers.6.mlp.gate_proj.weight",
    "model.layers.6.mlp.up_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
    "model.layers.6.input_layernorm.weight",
    "model.layers.6.post_attention_layernorm.weight",
    "model.layers.7.self_attn.q_proj.weight",
    "model.layers.7.self_attn.k_proj.weight",
    "model.layers.7.self_attn.v_proj.weight",
    "model.layers.7.self_attn.o_proj.weight",
    "model.layers.7.mlp.gate_proj.weight",
    "model.layers.7.mlp.up_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
    "model.layers.7.input_layernorm.weight",
    "model.layers.7.post_attention_layernorm.weight",
    "model.layers.8.self_attn.q_proj.weight",
    "model.layers.8.self_attn.k_proj.weight",
    "model.layers.8.self_attn.v_proj.weight",
    "model.layers.8.self_attn.o_proj.weight",
    "model.layers.8.mlp.gate_proj.weight",
    "model.layers.8.mlp.up_proj.weight",
    "model.layers.8.mlp.down_proj.weight",
    "model.layers.8.input_layernorm.weight",
    "model.layers.8.post_attention_layernorm.weight",
    "model.layers.9.self_attn.q_proj.weight",
    "model.layers.9.self_attn.k_proj.weight",
    "model.layers.9.self_attn.v_proj.weight",
    "model.layers.9.self_attn.o_proj.weight",
    "model.layers.9.mlp.gate_proj.weight",
    "model.layers.9.mlp.up_proj.weight",
    "model.layers.9.mlp.down_proj.weight",
    "model.layers.9.input_layernorm.weight",
    "model.layers.9.post_attention_layernorm.weight",
    "model.layers.10.self_attn.q_proj.weight",
    "model.layers.10.self_attn.k_proj.weight",
    "model.layers.10.self_attn.v_proj.weighQt"
    "model.layers.10.self_attn.o_proj.weight",
    "model.layers.10.mlp.gate_proj.weight",
    "model.layers.10.mlp.up_proj.weight",
    "model.layers.10.mlp.down_proj.weight",
    "model.layers.10.input_layernorm.weight",
    "model.layers.10.post_attention_layernorm.weight",
    "model.layers.11.self_attn.q_proj.weighQt"
    "model.layers.11.self_attn.k_proj.weight",
    "model.layers.11.self_attn.v_proj.weight",
    "model.layers.11.self_attn.o_proj.weight",
    "model.layers.11.mlp.gate_proj.weight",
    "model.layers.11.mlp.up_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
    "model.layers.11.input_layernorm.weight",
    "model.layers.11.post_attention_layernorm.weight",
    "model.layers.12.self_attn.q_proj.weight",
    "model.layers.12.self_attn.k_proj.weight",
    "model.layers.12.self_attn.v_proj.weight",
    "model.layers.12.self_attn.o_proj.weight",
    "model.layers.12.mlp.gate_proj.weight",
    "model.layers.12.mlp.up_proj.weighQt"
    "model.layers.12.mlp.down_proj.weight",
    "model.layers.12.input_layernorm.weight",
    "model.layers.12.post_attention_layernorm.weight",
    "model.layers.13.self_attn.q_proj.weight",
    "model.layers.13.self_attn.k_proj.weight",
    "model.layers.13.self_attn.v_proj.weight",
    "model.layers.13.self_attn.o_proj.weight",
    "model.layers.13.mlp.gate_proj.weight",
    "model.layers.13.mlp.up_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.13.input_layernorm.weight",
    "model.layers.13.post_attention_layernorm.weight",
    "model.layers.14.self_attn.q_proj.weight",
    "model.layers.14.self_attn.k_proj.weight",
    "model.layers.14.self_attn.v_proj.weight",
    "model.layers.14.self_attn.o_proj.weight",
    "model.layers.14.mlp.gate_proj.weight",
    "model.layers.14.mlp.up_proj.weight",
    "model.layers.14.mlp.down_proj.weight",
    "model.layers.14.input_layernorm.weighQt"
    "model.layers.14.post_attention_layernorm.weight",
    "model.layers.15.self_attn.q_proj.weight",
    "model.layers.15.self_attn.k_proj.weight",
    "model.layers.15.self_attn.v_proj.weight",
    "model.layers.15.self_attn.o_proj.weight",
    "model.layers.15.mlp.gate_proj.weight",
    "model.layers.15.mlp.up_proj.weight",
    "model.layers.15.mlp.down_proj.weight",
    "model.layers.15.input_layernorm.weight",
    "model.layers.15.post_attention_layernorm.weight",
    "model.norm.weight",
]
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

    for n, p in model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in h.modules)

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
    for p in model.parameters():
        if not p.requires_grad:
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

        for p in model.parameters():
            if not p.requires_grad:
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
        for d_batch in disruption_batches[-8:]:
            output = model(**d_batch)
            loss += cross_entropy_loss(output, d_batch)
        disr_loss = loss / len(disruption_batches[-8:])

    update_norm2 = update_norm ** 2
    logging.info(f"{wmdp_acc=:.4f} {disr_loss=:.4f} {update_norm2=:.4f}")
    wandb.log({"wmdp_acc": wmdp_acc, "disr_loss": disr_loss, "update_norm2": update_norm2})
    if wmdp_acc < 0.3 or disr_loss > 2.62:
        raise StopIteration


# _lr = 3e-3
_lr = 1e-2
# for modules, unlearning_rate in [
#     # (["up_proj"], _lr),
#     # (["down_proj"], _lr * 0.1),
#     # (["gate_proj"], _lr * 4),
#     # (["q_proj"], _lr),
#     # (["k_proj"], _lr * 4),
#     # (["v_proj"], _lr),
#     # (["o_proj"], _lr),
#     (["gate_proj", "o_proj", "v_proj", "up_proj"], _lr * 0.1),
# ]:
# for start_layer in range(0, 16, 4):

    # modules = [f".{num}.mlp.gate_proj" for num in range(start_layer, start_layer + 4)]
for modules in param_names:
    # construct hyperparams
    h = OmegaConf.create(
        dict(
            normalize_grads=False,
            unlearning_loss_fn="neg_cross_entropy",
            # unlearning_loss_fn="neg_entropy",
            # unlearning_loss_fn="correct_logit_minus_avg",
            #
            use_masking=False,
            unlearning_rate=_lr,
            modules=[modules],
            percentile=None,
        )
    )
    s.unlearning_epochs = 6

    name = f"{h.unlearning_rate}|{modules}"
    wandb.init(project="wmdp_single3", name=name, group="all_module_all_layers")
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


