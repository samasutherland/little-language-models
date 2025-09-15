import torch, os, sys
from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from functools import partial
from data.datasets import SimpleStoriesBPEDataset
from lib.model_layers.transformer import Transformer
import math

def pad_collate_fn(batch, pad_id):
    x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    m = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": m}

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    data = load_dataset("SimpleStories/SimpleStories")
    dataset = SimpleStoriesBPEDataset(data["train"], max_length=500)

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
    loader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate,
                        num_workers=4, persistent_workers=True)

    torch.set_default_dtype(torch.bfloat16)
    vocab_size = max(
        max(dataset.tok.vocab.keys(), default=-1),
        max(getattr(dataset.tok, "merge_dict", {}).values(), default=-1),
        dataset.pad_id,
        dataset.end_id,
    ) + 1
    model = Transformer(16, vocab_size=vocab_size, max_context=500, embedding_dim=768, n_heads=8, qk_dim=96, projection_dim=256).to(dtype=torch.bfloat16, device=device)

    model.embedding.weight.data = model.embedding.weight.data.to(torch.bfloat16)
    model.embedding.padding_idx = dataset.pad_id

    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    loss_steps, loss_values = [], []
    prompt_text = "The dog"

    resume_path = os.path.join(save_dir, "ckpt_latest.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(dtype=torch.bfloat16, device=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        for s in optimizer.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        try:
            hist = torch.load(os.path.join(save_dir, "loss_history.pt"), map_location="cpu")
            loss_steps = hist.get("steps", [])
            loss_values = hist.get("losses", [])
        except:
            pass

    def generate_sample(prompt, n_words=15, max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9):
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

    # total_steps = len(loader)
    #
    # def lr_lambda(s):
    #     p = s / total_steps
    #     return 0.5 * (1.0 + math.cos(math.pi * p))
    #
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=30)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, total_steps=5000)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", total=5000, file=sys.stdout, miniters=10)
        epoch_loss, n_batches = 0.0, 0
        for batch in pbar:
            if global_step % 10 == 0:
                try:
                    s = generate_sample(prompt_text, 15, 60, temperature=0.9, top_k=50, top_p=0.9)
                    tqdm.write(s, file=sys.stdout)
                except Exception as e:
                    tqdm.write(f"[gen err: {e}]", file=sys.stdout)
            if global_step % 500 == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step,
                            "epoch": epoch},
                           os.path.join(save_dir, f"ckpt_step_{global_step}.pt"))
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step,
                            "epoch": epoch},
                           os.path.join(save_dir, "ckpt_latest.pt"))
                torch.save({"steps": loss_steps, "losses": loss_values}, os.path.join(save_dir, "loss_history.pt"))

            x = batch["input_ids"].to(device, non_blocking=True)
            logits = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            loss.backward()
            loss_steps.append(global_step)
            loss_values.append(loss.item())
            optimizer.step()
            scheduler.step(loss.item())
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}, lr: {scheduler.get_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": global_step, "epoch": epoch}, os.path.join(save_dir, "ckpt_best.pt"))
        torch.save({"steps": loss_steps, "losses": loss_values}, os.path.join(save_dir, "loss_history.pt"))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
