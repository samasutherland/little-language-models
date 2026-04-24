# little Language Models (lLMs)
## Overview
This repo is where I am running a series of personal experiments to investigate architectural changes in small-scale language models.
The framework driven by pydantic, allowing experiments to be configured via a centralised configuration directory.
Modules are coupled to Factory classes, allowing extensible addition of components.
The main goal of this repository is for me to brush up on ML skills, gain experience with natural language processing, and satisfy my curiosity on how wacky ideas can affect performance.
Read more about the experiments at [my blog](https://samasutherland.github.io/lLMs/)

## Repository Layout
- `configs/`: Configuration files for managing experiments
- `scripts/`: Pretraining/training/posttraining entrypoints and utilities
- `lib/`: Library containing model/data/training components and implementations
- `notebooks/` - analysis and figures

## Training Workflow
1. **Pretraining stage** (`scripts/1-pretraining.py`)
   - Scales the number of layers in a model to achieve a desired ratio of tokens processed to model parameters
   - Chooses the best batch size to maximise GPU memory usage
   - Performs a learning rate sweep on a reduced runtime subset
   - Warning that this stages writes new parameters to the context.yaml configuration file.
2. **Training stage** (`scripts/2-training.py`)
   - Trains model for target time period
   - Logs metrics to Aim
   - Saves checkpoints
3. **Posttraining stage** (`scripts/3-posttraining.py`)
   - Evaluates the trained model on the validation/test set
   - Generates some example generations using test prompts defined in the script
   - Logs posttraining metrics/examples to Aim

## Quick Start
### Local training
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If you are running on a laptop or other system without a decent GPU, I recommend switching to the local_train branch, where the configs define a small transformer suitable for laptop training.
Then run the start_local script which performs pretraining, training, and then evaluates the model:
```bash
./scripts/start_local.sh
```
Training runs generate logs in logs/, model checkpoints in data/checkpoints, and aim runs in .aim/.

To run an optuna search over hyperparameters, use
```bash
./scripts/start_optuna.sh
```
Note that hyperparameter search variables are currently hard-coded into the script.

### Runpod Training
To train on runpod, you must first load the relevant *** please complete***

## Configuration Files
Each configuration file is responsible for its own relevant parameters:
- **data.yaml:** Parameters related to the dataset and dataloader
- **context.yaml:** Runtime-context parameters that are adjusted by the pretraining step including num_layers, descent_steps, and learning rate, as well as a couple of other parameters useful to define at runtime
- **model.yaml** Parameters that define the model structure
- **pretraining.yaml** Targets variables for pretraining, and the sweep parameters used to find them
- **server.yaml** Machine-specific variables used for specifying worker counts and memory options
- **training.yaml** Parameters that adjust the training loop used to train the models

