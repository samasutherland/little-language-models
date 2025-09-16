import torch

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