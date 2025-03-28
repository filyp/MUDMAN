import logging
from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

import wandb
from utils.loss_fns import *
from utils.training import *


def surgical_irreversible_unlearning(
    h,
    config,
    retain_batches,
    forget_batches,
    f_eval,
    r_eval,
    allowed_r_loss,
    model=None,
    # soft_threshold=None,
    eval_wmdp_every=None,
    allowed_mmlu_acc=None,
):
    h.fork_every_n_loops = int(h.fork_every_n_loops)
    # unlearn = True
    if eval_wmdp_every is not None:
        from utils.wmdp_eval import eval_on_wmdp
        from utils.mmlu_eval import eval_on_mmlu

    if model is None:
        if config.model_id in ["meta-llama/Llama-3.2-1B"]:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=pt.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_id)
    model.config.use_cache = False

    clip_at = h.additional_param if config.additional_param_name == "clip_at" else 0

    # get params to intervene on
    interven_params = [
        p
        for name, p in model.named_parameters()
        if any(f"{m}.weight" in name for m in config.target_modules)
    ]
    # normalizing by the total number of parameters ** 0.5 is useful for
    # experiments with varying target modules, to make unlearning rate comparable
    total_interven_numel = sum(p.numel() for p in interven_params)

    # require grads only on intervened params
    for p in model.parameters():
        p.requires_grad = id(p) in [id(p) for p in interven_params]

    for p in interven_params:
        p.retain_acc = pt.zeros_like(p.data)
        if config.additional_param_name == "forget_momentum":
            p.forget_acc = pt.zeros_like(p.data)
        p.base_data = p.data.clone().detach()

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    if config.additional_param_name == "rep_eng_retain_lr":
        frozen_model = deepcopy(model)
        frozen_model.eval()

    # ! unlearning loop
    passes_per_loop = (
        4
        + int(config.train_adversary)
        # note that now it's inefficient and uses two additional passes
        # but in principle could be optimized to reuse outputs and only need
        # additional forward pass on the frozen model
        + int(config.additional_param_name == "rep_eng_retain_lr")
    )
    _eval_counter = 0
    assert config.unlearn_steps % passes_per_loop == 0
    # _results = []
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        model.train()
        f_input_ids = next(forget_iter)
        r_input_ids = next(retain_iter)

        if loop_num % h.fork_every_n_loops == 0:
            for p in interven_params:
                p.adv_data = p.base_data.clone().detach()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        for p in interven_params:  # switch to base model
            p.data = p.base_data
        output = model(r_input_ids)
        loss = cross_entropy_loss(output, r_input_ids)
        if config.additional_param_name == "rep_eng_retain_lr":
            # ! representation engineering retain loss
            rep_eng_loss = circuit_breaker_retain_loss(
                model, r_input_ids, frozen_model, square_norm=config.square_norm
            )
            # note this loss is scaled both by this LR and retaining_rate
            rep_eng_loss *= h.additional_param
            loss += rep_eng_loss
        loss.backward()
        for p in interven_params:
            assert p.data.data_ptr() == p.base_data.data_ptr()
            # ! update disruption scores
            p.retain_acc *= h.retain_momentum
            p.retain_acc += p.grad * (1 - h.retain_momentum)
            # ! retain update
            p.base_data -= h.retaining_rate * p.retain_acc

        if not config.train_adversary:
            for p in interven_params:
                p.adv_data = p.base_data

        # if unlearn:
        # ! relearn the adversary
        model.zero_grad(set_to_none=True)
        for p in interven_params:  # switch to adversary
            p.data = p.adv_data
        output = model(f_input_ids)
        if config.train_adversary:
            loss = cross_entropy_loss(output, f_input_ids)
            loss.backward(retain_graph=True)
            for p in interven_params:
                assert p.data.data_ptr() == p.adv_data.data_ptr()
                # apply adversary update
                p.adv_data -= h.adv_lr * p.grad
                # decay adversary into base model
                p.adv_data *= h.adv_decay
                p.adv_data += p.base_data * (1 - h.adv_decay)

        # ! unlearning step with masking
        # get unlearning grads loss from adversary
        # reuse the computation graph from previous block
        model.zero_grad(set_to_none=True)
        loss_fn = loss_fns[config.unlearning_loss_fn]
        loss = loss_fn(output, f_input_ids, clip_at)
        loss.backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            assert p.data.data_ptr() == p.adv_data.data_ptr()

            if config.additional_param_name == "forget_momentum":
                p.forget_acc *= h.additional_param
                p.forget_acc += p.grad * (1 - h.additional_param)
                p.grad = p.forget_acc.clone().detach()

            if config.use_masking:
                mask = p.retain_acc.sign() == p.grad.sign()
                p.grad *= mask

            if config.additional_param_name == "discard_growing_weights":
                mask2 = p.base_data.sign() != p.grad.sign()
                p.grad[mask2] *= h.additional_param

            # normalize
            if config.normalize_grads:
                p.grad *= total_interven_numel**0.5 / grad_norm

            p.base_data -= h.unlearning_rate * p.grad

            if config.additional_param_name == "adv_update":
                assert config.train_adversary  # otherwise it may be wrong
                p.adv_data -= h.unlearning_rate * p.grad * h.additional_param

        # ! eval current loss
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done // 30 > _eval_counter:
            _eval_counter += 1
            for p in interven_params:  # switch to base model
                p.data = p.base_data
            res = eval_(
                model,
                f_eval,
                r_eval,
                allowed_r_loss,
                _passes_done,
                use_wandb=eval_wmdp_every is not None,
            )
            # _results.append(res | {"step": _passes_done})
            # if soft_threshold is not None and res["retain_loss"] > soft_threshold:
            #     if unlearn:
            #         logging.info("unlearning disabled")
            #     unlearn = False
            # else:
            #     if not unlearn:
            #         logging.info("unlearning enabled")
            #     unlearn = True

        if eval_wmdp_every is not None and _passes_done % eval_wmdp_every == 0:
            wmdp_acc = eval_on_wmdp(model)
            mmlu_acc = eval_on_mmlu(model)
            wandb.log(
                res | {"wmdp_accuracy": wmdp_acc, "mmlu_accuracy": mmlu_acc},
                step=_passes_done,
            )
            print(f"wmdp_acc={wmdp_acc}, mmlu_acc={mmlu_acc}")

            if allowed_mmlu_acc is not None and mmlu_acc < allowed_mmlu_acc:
                logging.info(f"Pruning trial because mmlu accuracy is too low")
                wandb.finish()
                raise optuna.TrialPruned()

    for p in interven_params:  # switch to base model
        p.data = p.base_data

    # return model, _results
    return model
