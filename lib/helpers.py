import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from data.datasets import SimpleStoriesBPEDataset
from lib.models import MODEL_REGISTRY
import time
from contextlib import nullcontext

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
    with open("experiment/model.toml", "rb") as f:
        model_cfg = tomllib.load(f)
    with open("experiment/data.toml", "rb") as f:
        data_cfg = tomllib.load(f)
    with open("experiment/training.toml", "rb") as f:
        train_cfg = tomllib.load(f)

    return model_cfg, data_cfg, train_cfg

def loopy_loader(loader):
    while True:
        for batch in loader:
            yield batch

def get_data_loader(data_cfg, train_cfg, split=None):
    data = load_dataset(data_cfg["dataset"])
    tokenizer_model_path = data_cfg["tokenizer_path"]
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"] if split is None else split], model_path=tokenizer_model_path, max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
    loader = loopy_loader(DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=8))

    vocab_size = dataset.vocab_size
    return dataset, loader, vocab_size

def create_model(model_cfg, vocab_size, device, dataset):
    model_cls = MODEL_REGISTRY[model_cfg["model"]]

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



