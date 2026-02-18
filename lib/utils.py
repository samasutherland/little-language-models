import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from data.datasets import SimpleStoriesBPEDataset
import time
from contextlib import nullcontext
import os
import weakref
import math
import functools
from sympy import divisors
import gc
from torch import nn

from lib.components import language_models

import tomllib

def generate_sample(model, dataset, device, prompt, n_words=15, max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    with torch.no_grad():
        ids0 = dataset.tok.encode(prompt)
        ids = torch.tensor(ids0, dtype=torch.long, device=device).unsqueeze(0)
        for i in range(max_new_tokens):
            logits = model(ids)[:, -1, :].float()
            if dataset.pad_id < logits.size(-1):
                logits[:, dataset.pad_id] = float("-inf")
            logits = logits / max(temperature, 1e-8)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if dataset.eos_id is not None and dataset.eos_id < logits.size(-1) and int(
                    next_id.item()) == dataset.eos_id:
                break
            ids = torch.cat([ids, next_id], dim=1)
            if len(dataset.tok.decode(ids[0].tolist()).split()) >= n_words:
                break
    model.train()
    return " ".join(dataset.tok.decode(ids[0].tolist()).split()[:n_words])

def pad_collate_fn(batch, pad_id):
    x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    m = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": m}

def load_configs():
    with open("configs/model.toml", "rb") as f:
        model_cfg = tomllib.load(f)
    with open("configs/data.toml", "rb") as f:
        data_cfg = tomllib.load(f)
    with open("configs/training.toml", "rb") as f:
        train_cfg = tomllib.load(f)

    return model_cfg, data_cfg, train_cfg

def loopy_loader(loader):
    while True:
        for batch in loader:
            yield batch

def get_data_loader(data_cfg, train_cfg, batch_size=None, split=None, shuffle=True, num_workers=None):
    if batch_size is None:
        batch_size = train_cfg["batch_size"]
    data = load_dataset(data_cfg["dataset"])
    tokenizer_model_path = data_cfg["tokenizer_path"]
    dataset = SimpleStoriesBPEDataset(
        data[data_cfg["split"] if split is None else split],
        model_path=tokenizer_model_path,
        max_length=data_cfg["max_length"]
    )

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
    if num_workers is None:
        num_workers = 8#os.cpu_count()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 1 else False,
        pin_memory=True,
        prefetch_factor=8 if num_workers > 1 else None
    )

    def _finalizer(dataloader):
        it = getattr(dataloader, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass

    class _LoopingWrapper:
        def __init__(self, dataloader):
            self._underlying_dataloader = dataloader
            self._gen = loopy_loader(dataloader)
        def __iter__(self):
            return self._gen

    import weakref
    loader = _LoopingWrapper(loader)
    weakref.finalize(loader, _finalizer, loader._underlying_dataloader)

    vocab_size = dataset.vocab_size
    return dataset, loader, vocab_size

def create_model(model_cfg, vocab_size, device, dataset):
    model_cls = getattr(language_models, model_cfg["model"])

    model_cfg["global"] = model_cfg["global"] | {"vocab_size": vocab_size, "padding_idx": dataset.pad_id}

    model = model_cls(model_cfg).to(device=device)

    return model

def get_step_info(model, device, loader, criterion, optimizer, scheduler, its_per_step, timer_start=5, total_steps=10, val_steps=0):
    token_count = 0
    amp_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if torch.amp.autocast_mode.is_autocast_available(device) else nullcontext()

    assert timer_start < total_steps
    assert total_steps > 0

    its = 0
    data_iter = iter(enumerate(loader))

    for i, batch in data_iter:
        x = batch["input_ids"].to(device=device, non_blocking=True)

        if i == timer_start:
            # torch.cuda.synchronize()
            start_time = time.perf_counter()

        with amp_ctx:
            logits = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        its += 1
        if its == its_per_step:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            its = 0
        if i == total_steps - 1:
            break
    # torch.cuda.synchronize()
    end_time = time.perf_counter()
    tokens_per_step = x[:, 1:].numel()
    time_per_step = (end_time - start_time)/(total_steps - timer_start)

    out_loss = loss.item()
    del x, logits, loss
    if val_steps > 0:
        with torch.no_grad():
            val_losses = []
            model.eval()
            its=0
            for i, batch in data_iter:
                x = batch["input_ids"].to(device=device, non_blocking=True)
                with amp_ctx:
                    logits = model(x[:, :-1])
                    loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
                val_losses.append(loss.item())
                its += 1
                if its == val_steps:
                    break
            model.train()
        return time_per_step, tokens_per_step, torch.mean(torch.tensor(val_losses))

    return time_per_step, tokens_per_step, out_loss



def fibonacci_search(func, func_args=(), func_kwargs=None, lower_bound=1, upper_bound=32):
    if func_kwargs is None:
        func_kwargs = {}
    @functools.lru_cache
    def func_(probe):
        print(f"evaluating {probe}")
        return func(probe, *func_args, **func_kwargs)


    fib_nums = [1,1]
    while fib_nums[-1] <= (upper_bound - lower_bound):
        fib_nums.append(fib_nums[-1] + fib_nums[-2])

    fib_nums = fib_nums[::-1][1:]
    fib_index = 0

    # Initial pass uses two evaluations
    probe_upper = lower_bound + fib_nums[fib_index] - 1
    probe_lower = lower_bound + fib_nums[fib_index + 1] - 1

    result_upper = func_(probe_upper)
    result_lower = func_(probe_lower)


    if result_upper < result_lower:
        lower_bound = probe_lower + 1
        probe_lower = probe_upper
        result_lower = result_upper
        flag = "UPPER"
    else:
        upper_bound = probe_upper - 1
        probe_upper = probe_lower
        result_upper = result_lower
        flag = "LOWER"

    fib_index += 1

    while upper_bound - lower_bound > 1:
        if flag == "UPPER":
            probe_upper = lower_bound + fib_nums[fib_index] - 1
            result_upper = func_(probe_upper)

        elif flag == "LOWER":
            probe_lower = lower_bound + fib_nums[fib_index + 1] - 1
            result_lower = func_(probe_lower)

        if result_upper < result_lower:
            lower_bound = probe_lower + 1
            probe_lower = probe_upper
            result_lower = result_upper
            flag = "UPPER"
        else:
            upper_bound = probe_upper - 1
            probe_upper = probe_lower
            result_upper = result_lower
            flag = "LOWER"

        fib_index += 1

    result_upper = func_(upper_bound)
    result_lower = func_(lower_bound)
    if result_upper < result_lower:
        return upper_bound
    else:
        return lower_bound

def find_batch_size(model_cfg, data_cfg, train_cfg):
    num_layers = model_cfg['global']['num_layers']

    print(f"Finding batch size for model with {num_layers} layers")

    print("getting dataset")
    data = load_dataset(data_cfg["dataset"])
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], model_path=data_cfg["tokenizer_path"], max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

    vocab_size = dataset.vocab_size
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = create_model(model_cfg, vocab_size, device, dataset)

    print(f"Using {device} device")


    # Define training params
    criterion = getattr(nn, train_cfg["loss"])(ignore_index=dataset.pad_id)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = getattr(torch.optim, train_cfg["optimizer"])(
        [{"params": decay, "weight_decay": train_cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg["base_lr"], **train_cfg["optimizer_kwargs"])

    scheduler = getattr(torch.optim.lr_scheduler, train_cfg["scheduler"])(optimizer, max_lr=train_cfg["base_lr"],
                                                           total_steps=train_cfg["total_steps"])

    # First scale batch size by 2 until OOM
    print("finding batch size...")
    accumulated_batch_size = train_cfg["accumulated_batch_size"]
    batch_sizes = list(divisors(accumulated_batch_size))[::-1]
    batch_sizes.insert(0, accumulated_batch_size * 2)


    for i, batch_size in enumerate(batch_sizes):
        try:
            its_per_step = (train_cfg["accumulated_batch_size"] // batch_size) + int(train_cfg["accumulated_batch_size"] % batch_size > 0)
            dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg, batch_size=batch_size, num_workers=0)
            time_per_step, tokens_per_step, loss = get_step_info(model, device, loader, criterion, optimizer, scheduler, its_per_step, timer_start=10, total_steps=110)

            print(f"Batch size {batch_size} passed")
            break

        except RuntimeError as e:
            print(f"Batch size {batch_size} causes OOM")
            if batch_size == 1:
                raise ValueError("Model too big. Batch size 1 causes OOM.")
            del loader
            gc.collect()

    del loader
    gc.collect()

    batch_size = batch_size // 2 # half it - was getting random OOM in the actual training run.

    print(f"Final batch size: {batch_size}")
    train_cfg["batch_size"] = batch_size

    print(f"Calculating step count")
    its_per_step = (train_cfg["accumulated_batch_size"] // batch_size) + int(
        train_cfg["accumulated_batch_size"] % batch_size > 0)

    dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg, batch_size=batch_size, num_workers=0)
    time_per_step, tokens_per_step, loss = get_step_info(model, device, loader, criterion, optimizer, scheduler,
                                                         its_per_step, timer_start=10, total_steps=110)

    del loader
    gc.collect()


    total_steps = int((train_cfg["training_time"] * 60) / time_per_step)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_tokens = int(tokens_per_step * total_steps)
    print(f"Estimated total steps: {total_steps}\n Estimated total tokens: {total_tokens} (tokens/param: {total_tokens / total_params:.2f})")

    train_cfg["total_steps"] = total_steps
    train_cfg["warmup_steps"] = max(10, total_steps // 100)

    return model_cfg, data_cfg, train_cfg, total_tokens / total_params
