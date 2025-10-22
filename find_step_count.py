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

def find_step_count(model_cfg, data_cfg, train_cfg):

    print("getting dataset")
    data = load_dataset(data_cfg["dataset"])
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], model_path=data_cfg["tokenizer_path"], max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

    vocab_size = dataset.vocab_size
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = create_model(model_cfg, vocab_size, device, dataset)

    print(f"Using {device} device")


    # Define training params
    criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](
        [{"params": decay, "weight_decay": train_cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg["base_lr"], **train_cfg["optimizer_kwargs"])

    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=train_cfg["base_lr"],
                                                           total_steps=train_cfg["total_steps"])

    print("Finding Step Count...")

    its_per_step = (train_cfg["accumulated_batch_size"] // train_cfg["batch_size"]) + (train_cfg["accumulated_batch_size"] % train_cfg["batch_size"] > 0)
    num_workers = os.cpu_count()
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=num_workers, persistent_workers=False)
    time_per_step, tokens_per_step, loss = get_step_info(model, device, loader, criterion, optimizer, scheduler, its_per_step, timer_start=10, total_steps=110)

    total_steps = int((train_cfg["training_time"] * 60) / time_per_step)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_tokens = int(tokens_per_step * total_steps)
    print(f"Estimated total steps: {total_steps}\n Estimated total tokens: {total_tokens} (tokens/param: {total_tokens / total_params:.2f})")

    train_cfg["total_steps"] = total_steps
    train_cfg["warmup_steps"] = max(10, total_steps // 100) # 1% of time in warmup

    return model_cfg, data_cfg, train_cfg, total_tokens / total_params


