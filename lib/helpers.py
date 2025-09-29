import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
from data.datasets import SimpleStoriesBPEDataset
from lib.models import MODEL_REGISTRY
import time

import tomllib

def generate_sample(model, dataset, device, prompt, n_words=15, max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9):
    def _filter(logits, top_k, top_p):
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[..., -1, None]] = float("-inf")
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            mask = cdf > top_p
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        return logits

    model.eval()
    with torch.no_grad():
        ids0 = dataset.tok.encode(prompt) or [dataset.pad_id]
        ids = torch.tensor(ids0, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = model(ids)[:, -1, :].float()
            if dataset.pad_id < logits.size(-1):
                logits[:, dataset.pad_id] = float("-inf")
            logits = logits / max(temperature, 1e-8)
            logits = _filter(logits, top_k, top_p)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if dataset.end_id is not None and dataset.end_id < logits.size(-1) and int(
                    next_id.item()) == dataset.end_id:
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

def get_data_loader(data_cfg, train_cfg):
    data = load_dataset(data_cfg["dataset"])
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate,
                        num_workers=12, persistent_workers=True, pin_memory=True, prefetch_factor=8)

    torch.set_default_dtype(torch.bfloat16)
    vocab_size = dataset.max_id + 1
    return dataset, loader, vocab_size

def create_model(model_cfg, vocab_size, device, dataset):
    model_cls = MODEL_REGISTRY[model_cfg["model"]]

    model_cfg["global"] = model_cfg["global"] | {"vocab_size": vocab_size}

    model = model_cls(model_cfg).to(dtype=torch.bfloat16, device=device)

    model.embedding.weight.data = model.embedding.weight.data.to(torch.bfloat16)
    model.embedding.padding_idx = dataset.pad_id
    return model

def get_step_info(model, device, loader, criterion, optimizer, scheduler, timer_start=5, total_steps=10):
    token_count = 0

    assert timer_start < total_steps
    assert total_steps > 0


    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(device=device, non_blocking=True)

        if i == timer_start:
            # torch.cuda.synchronize()
            start_time = time.perf_counter()

        logits = model(x[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i == total_steps - 1:
            break
    # torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("finished finding token count")
    tokens_per_step = x[:, 1:].numel()
    time_per_step = (end_time - start_time)/(total_steps - timer_start)
    return time_per_step, tokens_per_step, loss.item()



