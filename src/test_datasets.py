# %%
import time
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from utils.loss_fns import loss_fns
from utils.data_loading import _load_camel_bio_topics
from utils.loss_fns import correct_logit_minus_avg_loss, cross_entropy_loss, neg_entropy_loss
from utils.mmlu_eval import eval_on_mmlu
from utils.wmdp_eval import eval_on_wmdp

pt.set_default_device("cuda")

# %%

model_id = "meta-llama/Llama-3.2-3B"
# model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
context_len = 200
batch_size = 4


# %%
def yield_batches(dataset, format_fn):
    # shuffle in case multiple topics are used
    dataset = dataset.shuffle(seed=42)
    batch = []
    for ex in dataset:
        full_message = format_fn(ex)
        if not full_message:
            continue

        tokens = tokenizer(full_message, return_tensors="pt")["input_ids"].squeeze()
        # print(tokens.shape)
        # skip short examples, and truncate to context_len
        if len(tokens) < context_len:
            continue
        tokens = tokens[:context_len]
        batch.append(tokens)

        if len(batch) == batch_size:
            yield pt.stack(batch)
            batch = []


# %%
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)

topics = ["Virology", "Microbiology", "Biotechnology", "Genetics", "Biochemistry"]
# topics = "all"
batches = yield_batches(
    dataset=_load_camel_bio_topics(topics),
    format_fn=lambda ex: ex["message_1"] + "\n\n" + ex["message_2"],
)
# exp_name = f"camel-bio-{topics}"

# batches = yield_batches(
#     dataset=load_dataset("lapisrocks/pile-bio", split="train"),
#     format_fn=lambda ex: ex["txt_chunk"],
# )
# exp_name = "pile-bio"


lr = 0.1e-4
retain_lr = 1e-4
optimizer = pt.optim.SGD(model.parameters(), lr=lr)
steps_done = 0

# retain_batches = yield_batches(
#     dataset=load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"),
#     format_fn=lambda ex: ex["text"],
# )
topics = ["Paleontology", "Biogeography", "Botany", "Marine biology", "Biomechanics"]
retain_batches = yield_batches(
    dataset=_load_camel_bio_topics(topics),
    format_fn=lambda ex: ex["message_1"] + "\n\n" + ex["message_2"],
)

# loss_fn_name = "correct_logit_minus_avg"
# loss_fn_name = "neg_cross_entropy"
loss_fn_name = "neg_entropy"

wandb.init(
    project="mudman-dataset-test-3b",
    # group=variant_name,
    name=f"{lr}:{retain_lr}:{loss_fn_name}"
)


# %%
steps_per_loop = 10
start_time = time.time()
model.train()
loss_fn = loss_fns[loss_fn_name]
for _ in range(100):

    for i in range(steps_per_loop):
        batch = next(batches)
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
        print(f"grad_norm={grad_norm}\tloss={loss}", end="\t")
        # optimizer.param_groups[0]["lr"] = lr * grad_norm
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()

        # I thought maybe retaining helps, but looks like it doesn't
        retain_batch = next(retain_batches)
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        out = model(retain_batch)
        retain_loss = cross_entropy_loss(out, retain_batch)
        retain_loss.backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
        print(f"grad_norm={grad_norm}\tretain_loss={retain_loss}")
        optimizer.param_groups[0]["lr"] = retain_lr
        optimizer.step()

        # print(f"loss={loss}, retain_loss={retain_loss}")

        

    steps_done += steps_per_loop
    print(f"time taken: {time.time() - start_time:.2f}s")

    model.eval()
    model.zero_grad(set_to_none=True)
    pt.cuda.empty_cache()
    with pt.no_grad():
        wmdp_acc = eval_on_wmdp(model, temperature=1, subset=100)
        mmlu_acc = eval_on_mmlu(model, temperature=1, subset=100)
    print(f"time taken: {time.time() - start_time:.2f}s", end=" ")
    print(f"wmdp_acc={wmdp_acc}, mmlu_acc={mmlu_acc}")
    wandb.log({"wmdp_accuracy": wmdp_acc, "mmlu_accuracy": mmlu_acc}, step=steps_done)

# %%
wandb.finish()

# %%
