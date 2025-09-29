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

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}


def find_max_batch_size(model, dataset, device, criterion, optimizer, starting_size=1, safety_factor=0.2, collate=None):
    batch_size = starting_size

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate,
                        num_workers=0, persistent_workers=False)


    del loader, logits, loss, x
    torch.cuda.synchronize()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    no_data_mem_usage = torch.cuda.memory_allocated()
    batch = next(iter(loader))

    x = batch["input_ids"].to(device, non_blocking=True)

    # Do it a second time so that the memory usage in the forward and backward pass happens while the optimizer parameters have been allocated
    logits = model(x[:, :-1]) # Mem used in forward pass
    loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    loss.backward() # Mem used in backward pass
    optimizer.step() # Mem used by optimizer
    optimizer.zero_grad()

    peak_mem_usage = torch.cuda.max_memory_allocated()

    delta = peak_mem_usage - no_data_mem_usage

    print(f"fixed bytes: {no_data_mem_usage}")
    print(f"bytes per batch: {delta}")

    total_size = no_data_mem_usage + (delta * batch_size)
    if total_size > (available_memory - (safety_factor)):
        raise Exception("The starting batch size is too big")

    while total_size < (available_memory - (safety_factor)):
        batch_size *= 2
        total_size = no_data_mem_usage + (delta * batch_size)

    return batch_size // 2



def find_param_token_ratio(model_cfg, vocab_size, device, dataset, train_cfg, collate, ):
    model_cfg["global"]["embedding_dim"] = model_cfg["global"]["num_layers"] * 128
    model_cfg["global"]["feedforward_dim"] = model_cfg["global"]["num_layers"] * 128 * 4
    model_cfg["attention"]["n_heads"] = model_cfg["global"]["num_layers"]
    model = create_model(model_cfg, vocab_size, device, dataset)
    model.train()
    print("model created.")

    # Define training params
    criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)
    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])

    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg["warmup_steps"])
    # decay = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer,
    #                                                    T_max=train_cfg["total_steps"] - train_cfg["warmup_steps"],
    #                                                    **train_cfg["scheduler_kwargs"])
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay],
    #                                                   milestones=[train_cfg["warmup_steps"]])
    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=train_cfg["base_lr"],
                                                           total_steps=train_cfg["total_steps"])

    print("finding max batch size...")
    batch_size = find_max_batch_size(model, dataset, device, criterion, optimizer, starting_size=1, collate=collate)
    print(f"Max batch size: {batch_size}")
    train_cfg["batch_size"] = batch_size

    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=12, persistent_workers=False, pin_memory=True, prefetch_factor=8)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Computing throughput and effective number of tokens...")
    total_tokens, total_steps = find_token_count(model, device, loader, criterion, optimizer, scheduler)

    train_cfg["total_steps"] = total_steps

    print(
        f"Total tokens: {total_tokens}, total params: {total_params}, ratio: {total_tokens / total_params} (target 20)")

    del model, loader, optimizer, criterion, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return total_tokens / total_params

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

    print(f"Using {device} device")

    train_ratio = find_param_token_ratio(model_cfg, vocab_size, device, dataset, train_cfg, collate)

    print(f"training on {train_ratio:.2f} tokens per parameter.")

    model_cfg_path.write_text(dumps(model_cfg), encoding="utf-8")
    data_cfg_path.write_text(dumps(data_cfg), encoding="utf-8")
    train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")




if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
