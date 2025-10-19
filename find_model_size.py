import torch, os, sys
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import time
import copy
import gc
from lib.helpers import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model
from pathlib import Path
from tomlkit import parse, dumps
from functools import partial
from tqdm import tqdm
import torchinfo
from data.datasets import SimpleStoriesBPEDataset
from datasets import load_dataset
from find_step_count import find_step_count
from find_batch_size import find_batch_size

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}


print("loading configs...")
model_cfg_path = Path("experiment/model.toml")
data_cfg_path = Path("experiment/data.toml")
train_cfg_path = Path("experiment/training.toml")

model_cfg = parse(model_cfg_path.read_text(encoding="utf-8"))
data_cfg = parse(data_cfg_path.read_text(encoding="utf-8"))
train_cfg = parse(train_cfg_path.read_text(encoding="utf-8"))

def calculate_tokens_per_parameter(num_layers, model_cfg, data_cfg, train_cfg):
    model_cfg["global"]["num_layers"] = num_layers
    model_cfg, data_cfg, train_cfg, tokens_per_param = find_batch_size(model_cfg, data_cfg, train_cfg)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    # model_cfg, data_cfg, train_cfg, tokens_per_param = find_step_count(model_cfg, data_cfg, train_cfg)
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    # gc.collect()
    print(f"num layers {num_layers} gets {tokens_per_param} tokens per parameter.")
    return model_cfg, data_cfg, train_cfg, tokens_per_param

num_layers = 1
results = {}
while True:
    model_cfg, data_cfg, train_cfg, tokens_per_param = calculate_tokens_per_parameter(num_layers, model_cfg, data_cfg, train_cfg)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    results[num_layers] = (copy.deepcopy(model_cfg), copy.deepcopy(data_cfg), copy.deepcopy(train_cfg))
    if tokens_per_param < train_cfg["tokens_per_param"]:
        break
    num_layers *= 2

lower_bound = num_layers // 2
upper_bound = num_layers


while upper_bound - lower_bound > 1:
    num_layers = lower_bound + (upper_bound - lower_bound) // 2
    model_cfg, data_cfg, train_cfg, tokens_per_param = calculate_tokens_per_parameter(num_layers, model_cfg, data_cfg,
                                                                                      train_cfg)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    results[num_layers] = (copy.deepcopy(model_cfg), copy.deepcopy(data_cfg), copy.deepcopy(train_cfg))
    if tokens_per_param > train_cfg["tokens_per_param"]:
        lower_bound = num_layers
    else:
        upper_bound = num_layers

print(f"final num layers: {upper_bound}")
model_cfg, data_cfg, train_cfg = results[upper_bound] # prefer slightly overparameterised
model_cfg_path.write_text(dumps(model_cfg), encoding="utf-8")
data_cfg_path.write_text(dumps(data_cfg), encoding="utf-8")
train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")

