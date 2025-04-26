"""Evaluations for WMDP, MMLU, WMDP-Deduped."""
# %%
import torch as pt
from transformers import AutoTokenizer

from datasets import load_dataset

pt.set_default_device("cuda")
answer_tokens = [" A", " B", " C", " D"]


# %%
def load_wmdp_deduped(paths, task="wmdp-deduped"):
    base_url = "https://raw.githubusercontent.com/aghyad-deeb/unlearning_evaluation/refs/heads/main/data"
    return load_dataset(
        "json",
        data_files=[f"{base_url}/{task}/{path}.jsonl" for path in paths],
        split="train",
    )


def format_prompt(ex):
    # taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wmdp/_default_template_yaml
    # "The following are multiple choice questions (with answers) about biology."
    return f"""\
{ex["question"].strip()}
A. {ex["choices"][0]}
B. {ex["choices"][1]}
C. {ex["choices"][2]}
D. {ex["choices"][3]}
Answer:"""
    # Answer (just A, B, C or D):"""


# %% prepare filtered mmlu dataset
mmlu_dataset = load_dataset("cais/mmlu", "all", split="validation")

# filter out all the subcategories of biology and health
# keep even the ones like anatomy, clinical_knowledge and professional_medicine,
# because they contain some questions about molecular biology
categories_to_reject = {
    "college_biology",
    "college_medicine",
    "high_school_biology",
    "human_aging",
    "medical_genetics",
    "nutrition",
    "professional_medicine",
    "virology",
    "anatomy",
    "clinical_knowledge",
}
# filter out the ones in the categories_to_reject
filtered_mmlu = [ex for ex in mmlu_dataset if ex["subject"] not in categories_to_reject]


# %%


def eval_on(dataset_name, model, batch_size=16, subset=None, temperature=0):
    assert model.config.name_or_path in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]  # fmt: skip
    pt.cuda.empty_cache()

    dataset = dict(
        wmdp_bio=load_dataset("cais/wmdp", "wmdp-bio")["test"],
        wmdp_deduped_attack=load_wmdp_deduped(["split_0", "split_1", "split_2", "split_3"]),  # fmt: skip
        wmdp_deduped_eval=load_wmdp_deduped(["split_4"]),
        filtered_mmlu=filtered_mmlu,
    )[dataset_name]

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # note that this assumes start-of-sequence token is used (which is true for llama)
    answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)

    # sort wmdp_bio by the prompt length
    dataset = sorted(dataset, key=lambda ex: len(format_prompt(ex)))
    if subset is not None:
        dataset = dataset[:subset]

    acc = 0
    for i in range(0, len(dataset), batch_size):
        # print(i)
        batch = dataset[i : i + batch_size]
        batch_text = [format_prompt(ex) for ex in batch]

        input_dict = tokenizer(batch_text, return_tensors="pt", padding=True)

        with pt.inference_mode():
            output = model(**input_dict)
        last_positions = input_dict["attention_mask"].sum(dim=-1) - 1
        last_token_logits = output.logits[range(len(batch)), last_positions]

        probs = pt.softmax(last_token_logits, dim=-1)
        answer_probs = probs[:, answer_ids]
        # if not all(answer_probs.sum(dim=-1) > 0.2):
        #     raise ValueError("Sum of answer probs is too low")

        answer_probs /= answer_probs.sum(dim=-1, keepdim=True)  # normalize
        # assert pt.allclose(answer_probs.sum(dim=-1), pt.tensor(1.0, dtype=pt.bfloat16))
        _correct_answers = pt.tensor([ex["answer"] for ex in batch])

        if temperature == 1:
            correct_answer_probs = answer_probs[range(len(batch)), _correct_answers]
            acc += correct_answer_probs.sum().item()
        elif temperature == 0:
            # for temperature=0
            hits = answer_probs.argmax(dim=-1) == _correct_answers
            acc += hits.sum().item()
            # print(hits)
        else:
            raise ValueError(f"Not supported temperature: {temperature}")

        del answer_probs, probs, last_token_logits, output
        pt.cuda.empty_cache()

    return acc / len(dataset)


# %% notes on wmdp-bio
# note: all this was run on 75% of WMDP-Bio
# when using temperature=1
# Smol accuracy: 24.9%
# Llama accuracy: 32.8%

# when using temperature=0
# Smol accuracy: 24.8%
# Llama accuracy: 47.9%

# using bfloat16 is valid - it almost does not affect the accuracy

# ! interestingly, when using temperature=1, unlearning often increases the accuracy!
# that may be because it makes these probabilities more extreme?

# when using full dataset, rather than 75%, accuracy goes from
# 46.96% to 47.21%, so not much difference
# but retraining (even w/o unlearning) makes it go down to 42.02% (on full dataset)


# %% notes on mmlu
# Llama-3.2-1B
# full precision, temperature=0:   0.380143696930111         110s
# bfloat16, temperature=0:         0.37165251469627697       20s
# bfloat16, temperature=1:         0.30147779229261923

# other models are too stupid!
# SmolLM-135M
# full precision, temperature=0:   0.25146962769431747
# pythia-14m
# full precision, temperature=0:   0.2495101241018942
