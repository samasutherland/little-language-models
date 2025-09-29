import torch, os, sys
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import time
import copy
import gc
from lib.helpers import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model, get_step_info
from pathlib import Path
from tomlkit import parse, dumps
from functools import partial
from tqdm import tqdm
import torchinfo
from data.datasets import SimpleStoriesBPEDataset
from datasets import load_dataset

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}



print("loading configs...")
model_cfg_path = Path("experiment/model.toml")
data_cfg_path = Path("experiment/data.toml")
train_cfg_path = Path("experiment/training.toml")

model_cfg = parse(model_cfg_path.read_text(encoding="utf-8"))
data_cfg = parse(data_cfg_path.read_text(encoding="utf-8"))
train_cfg = parse(train_cfg_path.read_text(encoding="utf-8"))

print("getting dataset")
data = load_dataset(data_cfg["dataset"])
dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], max_length=data_cfg["max_length"])

collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

torch.set_default_dtype(torch.bfloat16)
vocab_size = max(
    max(dataset.tok.vocab.keys(), default=-1),
    max(getattr(dataset.tok, "merge_dict", {}).values(), default=-1),
    dataset.pad_id,
    dataset.end_id,
) + 1
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = create_model(model_cfg, vocab_size, device, dataset)

print(f"Using {device} device")


# Define training params
criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)
optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])

scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=train_cfg["base_lr"],
                                                       total_steps=train_cfg["total_steps"])

# First scale batch size by 2 until OOM
print("finding max learning rate...")
batch_size = 1
while True:
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                            num_workers=0, persistent_workers=False)
        get_step_info(model, device, loader, criterion, optimizer, scheduler, timer_start=0, total_steps=5)
        batch_size *= 2
        print(f"Batch size {batch_size} passed")
    except RuntimeError as e:
        print(f"Batch size to cause OOM: {batch_size}")
        break

lower_bound = batch_size // 2
upper_bound = batch_size

while upper_bound - lower_bound > 1:
    flag = True
    trial = lower_bound + (upper_bound - lower_bound) // 2
    try:
        loader = DataLoader(dataset, batch_size=trial, shuffle=True, collate_fn=collate,
                            num_workers=0, persistent_workers=False)
        get_step_info(model, device, loader, criterion, optimizer, scheduler, timer_start=0, total_steps=5)
    except RuntimeError as e:
        flag = False

    if flag:
        lower_bound = trial
    else:
        upper_bound = trial

print(f"Final batch size: {lower_bound}")
train_cfg["batch_size"] = batch_size
train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")





