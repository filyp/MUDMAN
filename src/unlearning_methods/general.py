import logging
from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import *


def unlearn(
    h,
    config,
    retain_batches,
    forget_batches,
):
    h.fork_every_n_loops = int(h.fork_every_n_loops)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=pt.bfloat16
    )
    model.config.use_cache = False

    clip_at = h.clip_at if "clip_at" in h else 0

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
        if "forget_momentum" in h:
            p.forget_acc = pt.zeros_like(p.data)
        p.base_data = p.data.clone().detach()

    # ! unlearning loop
    logging.info("step      base_f      base_r")
    if "rep_eng_retain_lr" in h:
        frozen_model = deepcopy(model)
        frozen_model.eval()

    # ! unlearning loop
    num_of_loops = int(len(forget_batches) * config.unlearning_epochs)
    for loop_num in range(num_of_loops):
        batch_index = loop_num % len(forget_batches)
        f_batch = forget_batches[batch_index]
        r_batch = retain_batches[batch_index]
        model.train()

        if loop_num % h.fork_every_n_loops == 0:
            for p in interven_params:
                p.adv_data = p.base_data.clone().detach()

        # ! retain pass
        model.zero_grad(set_to_none=True)
        for p in interven_params:  # switch to base model
            p.data = p.base_data
        output = model(**r_batch)
        retain_loss = cross_entropy_loss(output, r_batch["input_ids"])
        if "rep_eng_retain_lr" in h:
            # ! representation engineering retain loss
            rep_eng_loss = circuit_breaker_retain_loss(
                model, r_batch, frozen_model, square_norm=h.square_norm
            )
            # note this loss is scaled both by this LR and retaining_rate
            rep_eng_loss *= h.rep_eng_retain_lr
            retain_loss += rep_eng_loss
        retain_loss.backward()
        for p in interven_params:
            assert p.data.data_ptr() == p.base_data.data_ptr()
            # ! update disruption scores
            p.retain_acc *= h.retain_momentum
            p.retain_acc += p.grad * (1 - h.retain_momentum)
            # ! retain update
            p.base_data -= h.retaining_rate * p.retain_acc

        if not h.train_adversary:
            for p in interven_params:
                p.adv_data = p.base_data

        # ! relearn the adversary
        model.zero_grad(set_to_none=True)
        for p in interven_params:  # switch to adversary
            p.data = p.adv_data
        output = model(**f_batch)
        if h.train_adversary:
            adversary_loss = cross_entropy_loss(output, f_batch["input_ids"])
            adversary_loss.backward(retain_graph=True)
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
        pt.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        loss_fn = loss_fns[h.unlearning_loss_fn]
        forget_loss = loss_fn(output, f_batch["input_ids"], clip_at)
        forget_loss.backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in interven_params) ** 0.5
        for p in interven_params:
            assert p.data.data_ptr() == p.adv_data.data_ptr()

            if "forget_momentum" in h:
                p.forget_acc *= h.forget_momentum
                p.forget_acc += p.grad * (1 - h.forget_momentum)
                p.grad = p.forget_acc.clone().detach()

            if h.use_masking:
                mask = p.retain_acc.sign() == p.grad.sign()
                p.grad *= mask

            if "discard_growing_weights" in h:
                mask2 = p.base_data.sign() != p.grad.sign()
                p.grad[mask2] *= h.discard_growing_weights

            # normalize
            if h.normalize_grads:
                p.grad *= total_interven_numel**0.5 / grad_norm

            p.base_data -= h.unlearning_rate * p.grad

            if "adv_update" in h:
                assert h.train_adversary  # otherwise it may be wrong
                p.adv_data -= h.unlearning_rate * p.grad * h.adv_update

        # ! eval current loss
        logging.info(f"step {loop_num} \t retain_loss={retain_loss.item():.4f} \t forget_loss={forget_loss.item():.4f}")

    for p in interven_params:  # switch to base model
        p.data = p.base_data

    # return model, _results
    return model
