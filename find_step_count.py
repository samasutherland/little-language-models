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

def main():
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

    print("Finding Step Count...")

    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=8, persistent_workers=False)
    time_per_step, tokens_per_step, loss = get_step_info(model, device, loader, criterion, optimizer, scheduler, timer_start=10, total_steps=110)

    total_steps = (30 * 60) / time_per_step
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_tokens = int(tokens_per_step * total_steps)
    print(f"Estimated total steps: {total_steps}\n Estimated total tokens: {total_tokens} (tokens/param: {total_tokens / total_params}:.2f)")

    train_cfg["total_steps"] = total_steps
    train_cfg["warmup_steps"] = max(10, total_steps // 100) # 1% of time in warmup
    train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")

if __name__ == "__main__":
    main()

