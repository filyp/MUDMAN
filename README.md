# Installation

Clone the repository, create a virtual environment preferably with python3.12, and run:
```bash
pip install -r requirements.txt
```
(In case of problems, try running `pip install -r .pip_freeze.txt` instead, to install the exact tested package versions.)

# Running MUDMAN

```bash
python src/MUDMAN.py
```

It contains a simple example of unlearning on the Pile-Bio dataset with Llama-3.2-1B and then relearning back.

# Reproducing the paper results

In case of reproducibility problems not covered in the instructions below, feel free to sumbit an issue.

## Locally

To locally run a comparison of multiple methods in a single setup, run:

```bash
python src/study_runner.py --config-path configs/EXPERIMENT_NAME.yaml
```

To see available additional options, run `python src/study_runner.py --help`. For example you can run less trials, or choose which variant to use, which is useful for parallelization.

## Remotely

To run remotely with modal.com, first create a `secret.json` file in the repository root, containing database URL for Optuna, your HuggingFace token (to use Llama), and your Weights & Biases key:
```json
{
    "db_url": "postgresql://XXX",
    "hf_token": "hf_XXX",
    "wandb_key": "XXX"
}
```

Then run this script (it has the same options as `study_runner.py`):
```bash
modal run src/modal_runner.py --config-path configs/EXPERIMENT_NAME.yaml
```

## Experiment names for the Figure "Ablation study of MUDMAN"

- `configs/ablations_and_loss2,llama32,pile-bio.yaml`
- `configs/ablations_and_loss2,smol,pile-bio.yaml`
- `configs/ablations_and_loss2,pythia,pile-bio.yaml`
- `configs/ablations_and_loss2,llama32,python.yaml`
- `configs/ablations_and_loss2,smol,python.yaml`
- `configs/ablations_and_loss2,pythia,python.yaml`

On an Nvidia L40 GPU, experiments for one Llama-3.2-1B yaml should take around 5\*7h (5 methods inside), for SmolLM-135M 5\*9h, and for pythia-14m 5\*4h.

Then to visualize the results run `python src/plotting/ablations_and_loss.py`.

## Experiment names for the Figure "Accuracy on WMDP-Bio"

- `configs/wmdp7.yaml`

On an Nvidia L40 GPU, it should take around 5\*24h. (Note that this experiment only supports remote runs.)

Then to visualize the results run `python src/plotting/wmdp.py`.
