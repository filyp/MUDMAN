# note: there are some problems with running this
import logging

import torch as pt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from utils.loss_fns import circuit_breaker_forget_loss, circuit_breaker_retain_loss
from utils.training import eval_

# def compute_loss(
#     percent_done,
#     model,
#     forget_input_ids,
#     retain_input_ids,
#     target_layers,
#     retaining_rate,
#     unlearning_rate,
# ):

#     # Those are pretty much arbitrary, the important thing is that retain_coeff increases as the training progresses and forget_coeff decreases.
#     retain_coeff = retaining_rate * (percent_done / 2)
#     forget_coeff = unlearning_rate * (1 - percent_done / 2)

# if retain_coeff > 0:
#     retain_loss = circuit_breaker_retain_loss(model, retain_input_ids, LoRA=True)
# else:
#     retain_loss = 0

# if forget_coeff > 0:
#     forget_loss = circuit_breaker_forget_loss(
#         model, forget_input_ids, target_layers, LoRA=True
#     )
# else:
#     forget_loss = 0

# loss = retain_coeff * retain_loss + forget_coeff * forget_loss

# return loss


def circuit_breakers(
    h, config, retain_batches, forget_batches, f_eval, r_eval, allowed_f_loss
):
    logging.info(f"Running circuit breaker with params: {h}")

    model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # Add LoRA
    ret_lora_config = dict(lora_dropout=0.05, target_modules="all-linear")
    ret_lora_c = LoraConfig(r=16, **ret_lora_config)
    lora_model = get_peft_model(model, ret_lora_c, adapter_name="ret_lora", mixed=True)

    # get params to intervene on
    interven_params = [
        p
        for name, p in lora_model.named_parameters()
        if "ret_lora" in name
    ]
    total_interven_numel = sum(p.numel() for p in interven_params)

    num_layers = lora_model.config.num_hidden_layers
    target_layers = [num_layers // 2]

    retain_iter = iter(retain_batches)
    forget_iter = iter(forget_batches)

    passes_per_loop = 5
    for loop_num in range(config.unlearn_steps // passes_per_loop):
        r_input_ids = next(retain_iter)
        f_input_ids = next(forget_iter)

        model.zero_grad(set_to_none=True)

        retain_loss = circuit_breaker_retain_loss(
            model, r_input_ids, lora_model=lora_model
        )
        retain_loss.backward()
        # retain update
        for p in interven_params:
            if p.grad is not None:
                p.data -= h.retaining_rate * p.grad

        model.zero_grad(set_to_none=True)
        forget_loss = circuit_breaker_forget_loss(
            model, f_input_ids, target_layers, lora_model=lora_model
        )
        forget_loss.backward()
        grad_norm = (
            sum(p.grad.norm() ** 2 for p in interven_params if p.grad is not None)
            ** 0.5
        )
        for p in interven_params:
            if p.grad is not None:
                p.grad *= total_interven_numel**0.5 / grad_norm
                p.data -= h.unlearning_rate * p.grad

        # Evaluation
        _passes_done = (loop_num + 1) * passes_per_loop
        if _passes_done % 60 == 0:
            eval_(model, f_eval, r_eval, allowed_f_loss, _passes_done)

    return model
