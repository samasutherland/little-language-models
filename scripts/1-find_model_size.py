import copy
import gc
import torch
from lib.run_config import load_run_config, load_sweep_spec_for_model_size, persist_num_layers, persist_training_updates
from lib.data_config import build_dataset_and_loader
from lib.model_builder import build_model_from_config
from lib.utils import find_batch_size

print("Loading configs...")
run_config = load_run_config()
data_config = run_config.load_data()
base_train_config = run_config.load_training()
sweep = load_sweep_spec_for_model_size()

# Get dataset once for vocab_size and pad_id
dataset, _, vocab_size, pad_id = build_dataset_and_loader(
    data_config,
    base_train_config.batch_size,
    num_workers=0,
    loop=False,
)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using", device, "device")

target_tokens_per_param = sweep.target_tokens_per_param
initial_num_layers = sweep.initial_num_layers


def calculate_tokens_per_parameter(num_layers: int):
    ctx = run_config.get_build_context(vocab_size, pad_id, num_layers=num_layers)
    model = build_model_from_config(run_config.model_config_path, ctx=ctx)
    model = model.to(device=device)
    updated_train, tokens_per_param = find_batch_size(model, data_config, base_train_config, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    print(f"num layers {num_layers} gets {tokens_per_param} tokens per parameter.")
    return updated_train, tokens_per_param


num_layers = initial_num_layers
results = {}  # num_layers -> updated TrainingConfig
targ_dists = {}  # num_layers -> distance to target

while True:
    updated_train, tokens_per_param = calculate_tokens_per_parameter(num_layers)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    results[num_layers] = copy.deepcopy(updated_train)
    targ_dists[num_layers] = abs(target_tokens_per_param - tokens_per_param)
    if tokens_per_param < target_tokens_per_param:
        break
    num_layers *= 2

lower_bound = num_layers // 2
upper_bound = num_layers

while upper_bound - lower_bound > 1:
    num_layers = lower_bound + (upper_bound - lower_bound) // 2
    updated_train, tokens_per_param = calculate_tokens_per_parameter(num_layers)
    targ_dists[num_layers] = abs(target_tokens_per_param - tokens_per_param)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    results[num_layers] = copy.deepcopy(updated_train)
    if tokens_per_param > target_tokens_per_param:
        lower_bound = num_layers
    else:
        upper_bound = num_layers

best_num_layers = min(targ_dists, key=targ_dists.get)
print("final num layers:", best_num_layers)
best_train_config = results[best_num_layers]

persist_num_layers(run_config.model_config_path, best_num_layers)
persist_training_updates(
    run_config.training_config_path,
    batch_size=best_train_config.batch_size,
    total_steps=best_train_config.total_steps,
    warmup_steps=best_train_config.warmup_steps,
)
