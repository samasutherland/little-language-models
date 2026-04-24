# little Language Models (lLMs)
This repo is where I am running a series of personal experiments to investigate architectural changes in small-scale language models.
The framework is driven by pydantic, allowing experiments to be configured via a centralised configuration directory.
Modules are coupled to Factory classes, allowing extensible addition of components.
The main goal of this repository is for me to brush up on ML skills, gain experience with natural language processing, and satisfy my curiosity on how wacky ideas can affect performance.
Read more about the experiments at [my blog](https://samasutherland.github.io/lLMs/)

## Layout
- `configs/`: Configuration files for managing experiments
- `scripts/`: Pretraining/training/posttraining entrypoints and utilities
- `lib/`: Library containing model/data/training components and implementations
- `notebooks/` - Analysis and figures

## Training Workflow
1. **Pretraining** (`scripts/1-pretraining.py`)
   - Scales the number of layers in a model to achieve a desired ratio of tokens processed to model parameters
   - Chooses the best batch size to maximise GPU memory usage
   - Performs a learning rate sweep on a reduced runtime subset
   - Warning that this stage writes new parameters to the context.yaml configuration file.
2. **Training** (`scripts/2-training.py`)
   - Trains model for target time period
   - Logs metrics to Aim
   - Saves checkpoints
3. **Posttraining** (`scripts/3-posttraining.py`)
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
RunPod training is launched automatically by the GitHub Action in `.github/workflows/build_and_push_docker.yml`. If you would like to train some models via runpod you must do a few things first:
1. Fork this github repo.
2. Create a runpod account and add money. Generate and note down an API key.
3. Create a RunPod template, set the start command to `bash -lc ./scripts/start.sh`. You can use mine if you like: `9m9g78tjqi`
4. I use gdrive to store the results. If you would also like to use this, add `RUNPOD_SECRET_RCLONE_GDRIVE_CONF` to the runpod template. This value should be your base64-encoded `rclone.conf`, and is decoded to `/secrets/rclone.conf` in the container. Otherwise, change start.sh to use some other network storage.
5. Create a dockerhub account. Note down the access token.
6. Add necessary secrets to the github repo:
   1. `DOCKERHUB_TOKEN`: Docker Hub access token with permission to push images.
   2. `RUNPOD_API_KEY`: RunPod API key with permission to list/create pods.
7. Add necessary variables to the github repo:
   1. `DOCKERHUB_USERNAME`: Your Docker Hub username/namespace.
   2. `RUNPOD_TEMPLATE_ID`: The RunPod template ID used when creating the pod.

The github action workflow then has two jobs:
1. **`build-push`**
   - Logs into Docker Hub.
   - Builds the image from `dockerfile`.
   - Pushes two tags:
     - `docker.io/<DOCKERHUB_USERNAME>/model_trainer:latest`
     - `docker.io/<DOCKERHUB_USERNAME>/model_trainer:sha-<commit-sha>`
   - Exposes the SHA-tagged image as an output for the next job.
   - Converts the latest commit message into a RunPod-safe pod name.

2. **`launch-pod`**
   - Performs an auth preflight call to RunPod (`GET /v1/pods`).
   - Creates a new secure 1x RTX 4090 pod using your template ID.
   - Injects environment variables into the pod:
     - `GIT_SHA`
     - `IMAGE_TAG`
     - `POD_NAME`
   - The container runs `scripts/start.sh`, which:
     - runs pretraining, training, and posttraining,
     - archives `.aim`,
     - uploads logs/configs/artifacts via `rclone`,
     - removes the pod when complete.

This action is triggered whenever code is pushed to `train`, so be careful pushing to that branch or you could cost yourself money!

## Configuration Files
Each configuration file is responsible for its own relevant parameters:
- **data.yaml:** Parameters related to the dataset and dataloader
- **context.yaml:** Runtime-context parameters that are adjusted by the pretraining step including num_layers, descent_steps, and learning rate, as well as a couple of other parameters useful to define at runtime
- **model.yaml** Parameters that define the model structure
- **pretraining.yaml** Targets variables for pretraining, and the sweep parameters used to find them
- **server.yaml** Machine-specific variables used for specifying worker counts and memory options
- **training.yaml** Parameters that adjust the training loop used to train the models

