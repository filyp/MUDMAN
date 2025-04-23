# %%
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from utils.data_loading import _load_camel_bio_topics
from utils.loss_fns import cross_entropy_loss
from utils.mmlu_eval import eval_on_mmlu
from utils.wmdp_eval import eval_on_wmdp

pt.set_default_device("cuda")

# %%

model_id = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=pt.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
context_len = 200
batch_size = 4


# %%
def yield_batches(dataset, format_fn):
    batch = []
    for ex in dataset:
        full_message = format_fn(ex)

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

# # topics = ["Virology"]
# topics = "all"
# batches = yield_batches(
#     dataset=_load_camel_bio_topics(topics),
#     format_fn=lambda ex: ex["message_1"] + "\n\n" + ex["message_2"],
# )
# exp_name = f"camel-bio-{topics}"

batches = yield_batches(
    dataset=load_dataset("lapisrocks/pile-bio", split="train"),
    format_fn=lambda ex: ex["txt_chunk"],
)
exp_name = "pile-bio"


lr = 1e-4
optimizer = pt.optim.SGD(model.parameters(), lr=lr)
steps_done = 0

wandb.init(
    project="mudman-dataset-test",
    # group=variant_name,
    name=f"{lr}:{exp_name}",
)
# without training:
# wmdp_acc=0.615082482325216, mmlu_acc=0.5276497695852534
wandb.log({"wmdp_accuracy": 0.61508, "mmlu_accuracy": 0.52764}, step=steps_done)

# %%
for _ in range(10):

    for i in range(100):
        batch = next(batches)

        model.train()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()

        out = model(batch)
        loss = cross_entropy_loss(out, batch)
        loss.backward()
        optimizer.step()
    steps_done += 100

    wmdp_acc = eval_on_wmdp(model, temperature=0)
    mmlu_acc = eval_on_mmlu(model, temperature=0)
    print(f"wmdp_acc={wmdp_acc}, mmlu_acc={mmlu_acc}")
    wandb.log({"wmdp_accuracy": wmdp_acc, "mmlu_accuracy": mmlu_acc}, step=steps_done)

# %%

wandb.finish()