# %%
import zipfile
from pathlib import Path

import torch as pt
from huggingface_hub import hf_hub_download

from datasets import IterableDataset, IterableDatasetDict, load_dataset

# %%
# dataset = load_dataset("camel-ai/biology", split="train")  # this is really slow

path = hf_hub_download(
    repo_id="camel-ai/biology",
    repo_type="dataset",
    filename="biology.zip",
    # local_dir="datasets/",
    # local_dir_use_symlinks=False,
)
print(path)

# Get the path to the downloaded file
zip_path = Path(path)
# Create an extraction directory in the same parent folder
extract_dir = zip_path.parent / "extracted"
extract_dir.mkdir(exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# Find the extracted data file(s)
# Assuming there's a JSON file with the data
l_str = [str(path) for path in extract_dir.glob("**/*.json")]
# create a dataset from the extracted data
dataset = load_dataset("json", data_files=l_str, split="train")

# %%
# topics = dataset.unique("topic;")
# as estimated by Claude 3.7 Sonnet in relation to bio-terrorism:
topic_risk_ratings = {
    "Microbiology": 5,
    "Biotechnology": 5,
    "Virology": 5,
    "Genetics": 5,
    "Biochemistry": 4,
    "Immunology": 3,
    "Mycology": 3,
    "Parasitology": 3,
    "Cell biology": 3,
    "Biostatistics": 2,
    "Physiology": 2,
    "Entomology": 2,
    "Evolution": 2,
    "Ecology": 2,
    "Biophysics": 2,
    "Zoology": 1,
    "Neurobiology": 1,
    "Taxonomy": 1,
    "Anatomy": 1,
    "Biomechanics": 1,
    "Paleontology": 1,
    "Botany": 1,
    "Marine biology": 1,
    "Endocrinology": 1,
    "Biogeography": 1,
}

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "roneneldan/TinyStories-33M"
model = AutoModelForCausalLM.from_pretrained(model_id)
# %%
subset = dataset.filter(lambda x: x["topic;"] == "Virology")
# %%
ex = subset[11]
print(ex["message_1"])
print(ex["message_2"])


# %%
# Concatenate message_1 and message_2
full_message = ex["message_1"] + "\n" + ex["message_2"]

# Tokenize the message
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(full_message, return_tensors="pt")
input_ids = inputs.input_ids

# Get model predictions
with pt.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Calculate probabilities for each token
# For each position, we want the probability of the actual next token
probs = []
for i in range(input_ids.shape[1] - 1):
    next_token_id = input_ids[0, i + 1].item()
    next_token_logits = logits[0, i, :]
    next_token_probs = pt.softmax(next_token_logits, dim=0)
    prob = next_token_probs[next_token_id].item()
    probs.append(prob)

# Add a dummy probability for the last token (no next token to predict)
probs.append(1.0)

# Decode tokens individually for display
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Print full message with improbable tokens highlighted
from IPython.display import HTML, display
from html import escape

# Create HTML visualization with intensity based on improbability
html_result = ""
for token, prob in zip(tokens, probs):
    # Convert token ID back to text
    token_text = tokenizer.convert_tokens_to_string([token])
    
    # Calculate opacity based on improbability
    opacity = max(0, 0.1 - prob) * 7
    
    # Use a single highlight color (yellow) with varying opacity
    html_result += f"<span style='background-color: rgba(255, 255, 0, {opacity})'>{escape(token_text)}</span>"

# Display result with highlight intensity based on improbability
display(HTML(html_result))

# ... existing code ...