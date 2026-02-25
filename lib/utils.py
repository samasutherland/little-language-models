import torch
from torch.utils.data import DataLoader
from functools import partial
import time
from contextlib import nullcontext
import os
import math
import functools
from sympy import divisors
import gc
from torch import nn

from lib.data_config import DataConfig, build_dataset_and_loader
from lib.training_config import (
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    build_criterion,
)

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

def loopy_loader(loader):
    while True:
        for batch in loader:
            yield batch


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

def find_batch_size(
    model: nn.Module,
    data_config: DataConfig,
    train_config: TrainingConfig,
    device: str,
) -> tuple[TrainingConfig, float]:
    """
    Find max feasible batch size (then halve for safety), compute total_steps and warmup_steps.
    Returns (updated TrainingConfig, tokens_per_param).
    """
    num_layers = len(model.transformer_stack) if hasattr(model, "transformer_stack") else 0
    print(f"Finding batch size for model with {num_layers} layers")
    print("Getting dataset...")

    optimizer = build_optimizer(train_config, model)
    accumulated = train_config.accumulated_batch_size
    batch_sizes = list(divisors(accumulated))[::-1]
    batch_sizes.insert(0, accumulated * 2)

    batch_size = accumulated  # fallback
    dataset = None
    for bs in batch_sizes:
        try:
            dataset, loader, vocab_size, pad_id = build_dataset_and_loader(
                data_config, bs, num_workers=0, loop=False
            )
            criterion = build_criterion(train_config, ignore_index=pad_id)
            its_per_step = (accumulated // bs) + int(accumulated % bs > 0)
            scheduler = build_scheduler(train_config, optimizer, its_per_step=its_per_step)
            time_per_step, tokens_per_step, loss = get_step_info(
                model, device, loader, criterion, optimizer, scheduler,
                its_per_step, timer_start=10, total_steps=110,
            )
            print(f"Batch size {bs} passed")
            batch_size = bs
            del loader
            gc.collect()
            break
        except RuntimeError as e:
            print(f"Batch size {bs} causes OOM")
            if bs == 1:
                raise ValueError("Model too big. Batch size 1 causes OOM.") from e
            try:
                del loader
            except NameError:
                pass
            gc.collect()

    gc.collect()
    batch_size = batch_size // 2  # half for safety
    print(f"Final batch size: {batch_size}")

    its_per_step = (accumulated // batch_size) + int(accumulated % batch_size > 0)
    if dataset is None:
        dataset, loader, _, _ = build_dataset_and_loader(
            data_config, batch_size, num_workers=0, loop=False
        )
    else:
        _, loader, _, _ = build_dataset_and_loader(
            data_config, batch_size, num_workers=0, loop=False
        )
    criterion = build_criterion(train_config, ignore_index=dataset.pad_id)
    scheduler = build_scheduler(train_config, optimizer, its_per_step=its_per_step)
    time_per_step, tokens_per_step, loss = get_step_info(
        model, device, loader, criterion, optimizer, scheduler,
        its_per_step, timer_start=10, total_steps=110,
    )
    del loader
    gc.collect()

    total_steps = int((train_config.training_time * 60) / time_per_step)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_tokens = int(tokens_per_step * total_steps)
    tokens_per_param = total_tokens / total_params
    print(f"Estimated total steps: {total_steps}\n Estimated total tokens: {total_tokens} (tokens/param: {tokens_per_param:.2f})")

    updated = train_config.model_copy(
        update={
            "batch_size": batch_size,
            "total_steps": total_steps,
            "warmup_steps": max(10, total_steps // 100),
        }
    )
    return updated, tokens_per_param
