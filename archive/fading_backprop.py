# %%
import time
from copy import deepcopy

import torch as pt
import wandb
from utils import forward, get_perplexity, get_norm_of_weights_change, normal_train_step


# %%
def install_hooks_for_saving_gradients(model):
    # save the gradients downstream of each mlp, to be used in custom unlearning
    def save_output_grad_hook(module, grad_input, grad_output):
        module.output_grad = grad_output[0]
        # ! note that .detach().clone() may be needed too

    for layer in model.model.layers:
        assert not layer.mlp.down_proj._backward_hooks
        layer.mlp.down_proj.register_full_backward_hook(save_output_grad_hook)


def install_hooks_for_fading_backprop(model):
    # (tested for gemma-2-2b and Qwen2.5-0.5B)
    def scale_grad_hook(module, grad_input, grad_output):
        grad = list(grad_input)
        if grad[0] is None:
            # this happens on layer 0, with requires_grad=False on 1st MLP layer
            return
        # we rely on fade_factor set with set_fade_factor
        grad[0] *= module.fade_factor
        return grad

    for layer in model.model.layers:
        assert not layer.mlp._backward_hooks
        layer.mlp.register_full_backward_hook(scale_grad_hook)
        assert not layer.input_layernorm._backward_hooks
        layer.input_layernorm.register_full_backward_hook(scale_grad_hook)


def set_fade_factor(model, fade_factor):
    # saves the fade_factor in each relevant module, so that hooks can access it
    for layer in model.model.layers:
        layer.mlp.fade_factor = fade_factor
        layer.input_layernorm.fade_factor = fade_factor


# %%
def activation_agnostic(model, batch, lr):
    optimizer = pt.optim.SGD(model.parameters(), lr=lr)

    loss = forward(model, batch)
    loss.backward()
    # get rid of the normal grad
    optimizer.zero_grad(set_to_none=True)

    # calculate custom grads
    for layer in model.model.layers:
        module = layer.mlp.down_proj
        # get the downstream gradient saved by the hook
        output_grad = module.output_grad
        output_grad = output_grad[:, :-1, :]  # last token position has no gradient
        assert output_grad.norm(dim=-1).all()

        # calculate the projection
        projection_amps = pt.einsum("oi,bto->bti", module.weight, output_grad)
        projection_amps /= output_grad.norm(dim=-1, keepdim=True)
        update = pt.einsum("bto,bti->oi", output_grad, projection_amps)
        module.weight.grad = update

    optimizer.step()


name_to_function = dict(
    activation_agnostic=activation_agnostic,
)


# %%
def unlearn_and_relearn(
    model,
    forget_dataset,
    retain_dataset,
    wandb_group="default",
    unlearning_function="activation_agnostic",
    f_schedule="lambda step: 0",
    num_unlearning_steps=20,
    num_relearning_steps=10,
    eval_every_n_steps=2,
    relearn_lr=0.0003,
    unlearn_lr=1,
    pl_ppl_threshold=float("inf"),
    batch_size=32,
    allowed_retain_ppl_multiplier=1.2,
):
    # prepare wandb run
    wandb.init(
        project="fading_backprop",
        config={k: v for k, v in locals().items() if isinstance(v, (int, float, str))},
        group=wandb_group,
    )
    f_schedule = eval(f_schedule)
    unlearning_function = name_to_function[unlearning_function]

    original_state_dict = deepcopy(model.state_dict())

    install_hooks_for_saving_gradients(model)
    install_hooks_for_fading_backprop(model)

    # create dataset iterators
    forget_unlearn_iter = iter(forget_dataset["unlearn"].batch(batch_size))
    forget_relearn_iter = iter(forget_dataset["relearn"].batch(batch_size))
    retain_relearn_iter = iter(retain_dataset["relearn"].batch(batch_size))

    # initial perplexities
    ppl = {
        "forget": get_perplexity(model, forget_dataset),
        "retain": get_perplexity(model, retain_dataset),
    }
    print("initial perplexity: ", {k: f"{v:.2f}" for k, v in ppl.items()})

    # perplexity on retain set is not allowed to raise above this value
    # - this triggers relearning on retain set
    allowed_retain_perplexity = ppl["retain"] * allowed_retain_ppl_multiplier
    print(f"{allowed_retain_perplexity=:.2f}")

    # unlearning loop
    for step_num in range(num_unlearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        # retain set perplexity too high, so relearn it
        if ppl["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            set_fade_factor(model, 1)
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        # do a bit of forget relearning, to make unlearning more relevant
        elif ppl["forget"] > pl_ppl_threshold:
            print("**relearning forget", end="  ")
            # we intentionally use forget_UNlean_iter, to not affect relearn split here
            set_fade_factor(model, 1)
            normal_train_step(model, next(forget_unlearn_iter), relearn_lr)
        # unlearn forget
        else:
            print("  unlearning forget", end="  ")
            set_fade_factor(model, f_schedule(step_num))
            unlearning_function(model, next(forget_unlearn_iter), unlearn_lr)

        # evaluate
        if (step_num + 1) % eval_every_n_steps == 0:
            ppl = {
                "forget": get_perplexity(model, forget_dataset),
                "retain": get_perplexity(model, retain_dataset),
                "w_delta": get_norm_of_weights_change(model, original_state_dict),
            }
            print({k: f"{v:.2f}" for k, v in ppl.items()}, end="  ")
            wandb.log(ppl)

    # relearning loop
    print("\n### relearning started ###", end="  ")
    for step_num in range(num_relearning_steps):
        print(f"\nstep {step_num + 1:3}", end="  ")
        if ppl["retain"] > allowed_retain_perplexity:
            print("> relearning retain", end="  ")
            set_fade_factor(model, 1)
            normal_train_step(model, next(retain_relearn_iter), relearn_lr)
        else:
            print("  relearning forget", end="  ")
            set_fade_factor(model, 1)
            normal_train_step(model, next(forget_relearn_iter), relearn_lr)

        # evaluate
        if (step_num + 1) % eval_every_n_steps == 0:
            ppl = {
                "forget": get_perplexity(model, forget_dataset),
                "retain": get_perplexity(model, retain_dataset),
                "w_delta": get_norm_of_weights_change(model, original_state_dict),
            }
            print({k: f"{v:.2f}" for k, v in ppl.items()}, end="  ")
            wandb.log(ppl)

    print("\n###### relearning finished ######")
    wandb.finish()
    return ppl
