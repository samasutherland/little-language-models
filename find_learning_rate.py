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
dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], model_path=data_cfg["tokenizer_path"], max_length=data_cfg["max_length"])

collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

vocab_size = dataset.vocab_size
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")


# Define training params
criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)


peak_frac = train_cfg["warmup_steps"] / train_cfg["total_steps"]
print(f"peak_frac: {peak_frac}")

train_steps = train_cfg["total_steps"] // train_cfg["training_time"]
train_steps = max(train_steps, train_cfg["warmup_steps"] * 2)
print(f"train_steps: {train_steps}, warmup_steps: {train_cfg['warmup_steps']}")
if train_steps < train_cfg["warmup_steps"] * 2:
    print("warmup steps less than half trian steps - inaccurate results to follow!")

# This should take 10 minutes
print("finding best learning rate...")
min = -5
max = -1
lrs = torch.logspace(min, max, 10)

losses = []
last_loss = torch.inf
for lr in lrs:
    torch.manual_seed(42)
    model = create_model(model_cfg, vocab_size, device, dataset)
    num_workers = os.cpu_count()
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=num_workers, persistent_workers=False)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](
        [{"params": decay, "weight_decay": train_cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr, **train_cfg["optimizer_kwargs"])
    its_per_step = (train_cfg["accumulated_batch_size"] // train_cfg["batch_size"]) + int(train_cfg["accumulated_batch_size"] % train_cfg["batch_size"] > 0)
    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=lr,
                                                           total_steps=train_cfg["total_steps"] // its_per_step, pct_start=peak_frac,
                                                           div_factor=3., final_div_factor=10.)

    _, _, loss = get_step_info(model, device, loader, criterion, optimizer, scheduler, its_per_step, timer_start=0, total_steps=train_steps, val_steps=20)
    print(f"lr {lr} achieved loss {loss}")
    losses.append(loss)
    if loss > last_loss:
        break
    last_loss = loss


losses = torch.tensor(losses, dtype=torch.float32)
mask = torch.isfinite(losses)
if not torch.any(mask):
    raise RuntimeError("All LR trials produced non-finite losses.")
best_lr = float(lrs[torch.where(mask)[0][torch.argmin(losses[mask])].item()].item())

print(f"Best learning rate: {best_lr}")
train_cfg["base_lr"] = best_lr
train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")





