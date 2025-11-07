import torch, os, sys
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm.auto import tqdm
from aim import Run, Text
import time
import math
import tomllib
import uuid
from lib.utils import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model
from contextlib import nullcontext

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}
torch.manual_seed(42)

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    amp_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if torch.amp.autocast_mode.is_autocast_available(device) else nullcontext()

    print(f"Using {device} device")

    model_cfg, data_cfg, train_cfg = load_configs()
    dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg)
    _, val_loader, _ = get_data_loader(data_cfg, train_cfg, split="test")
    model = create_model(model_cfg, vocab_size, device, dataset)
    model.train()

    # Define training params
    criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)


    save_dir = "configs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    best_val_loss = float("inf")

    # optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg["warmup_steps"])
    # decay = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, T_max=train_cfg["total_steps"] - train_cfg["warmup_steps"], **train_cfg["scheduler_kwargs"])
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[train_cfg["warmup_steps"]])

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]]([{"params": decay, "weight_decay": train_cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg["base_lr"] / 3, **train_cfg["optimizer_kwargs"])
    peak_frac = train_cfg["warmup_steps"] / train_cfg["total_steps"]
    print(f"peak_frac: {peak_frac}")
    its_per_step = (train_cfg["accumulated_batch_size"] // train_cfg["batch_size"]) + int(train_cfg["accumulated_batch_size"] % train_cfg["batch_size"] > 0)
    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=train_cfg["base_lr"], total_steps=train_cfg["total_steps"] // its_per_step, pct_start=peak_frac, div_factor=3., final_div_factor=10.)

    experiment_name = f"{model_cfg['model']}:{model_cfg['global']['transformer_layer']}:{model_cfg['global']['attention_layer']}:{model_cfg['global']['activation']}"

    run = Run(experiment = experiment_name)
    run["model_cfg"] = model_cfg
    run["data_cfg"] = data_cfg
    run["train_cfg"] = train_cfg

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["param_count"] = total_params
    try:
        run["data_folder"] = os.environ["RUNPOD_POD_ID"]
    except KeyError:
        run["data_folder"] = uuid.uuid4().hex

    import signal, threading
    stop = threading.Event()
    def _handle_sigterm(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, _handle_sigterm)
    token_count = 0

    loss_buffer = torch.ones(100) * torch.inf
    def push_to_buffer(x):
        return torch.cat((loss_buffer[1:], torch.tensor([x])))


    its = 0
    step_scheduler = True

    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(device, non_blocking=True)
        token_count += x[:, 1:].numel()
        with amp_ctx:
            logits = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        # skip if loss is huge
        if loss.item() > 10 * torch.median(loss_buffer):
            print("loss huge. skipping...")
            continue

        loss_buffer = push_to_buffer(loss.item())

        if step_scheduler:
            current_lr = scheduler.get_last_lr()[0]
        run.track(loss.item(), name="loss", step=i, context={"subset": "train"})
        if not torch.isfinite(loss):
            pass
        else:
            run.track(torch.exp(loss.detach().to(torch.float32).clamp(max=88.72)).item(), name="perplexity", step=i, context={"subset": "train"})
        run.track(current_lr, name="lr", step=i, context={"subset": "train"})


        print(f"Step: {i}, LR: {current_lr}, loss: {loss.item()}")


        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": i}, os.path.join(save_dir, "ckpt_best.pt"))
            with open(os.path.join(save_dir, "best_loss_step.txt"), "w") as f:
                f.write(f"loss of {best_loss} achieved on step {i}")
        run.track(best_loss, name="best_loss", step=i, context={"subset": "train"})

        loss = loss / its_per_step

        loss.backward()
        its += 1
        if its == its_per_step:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step_scheduler:
                try:
                    scheduler.step()
                except ValueError:
                    print("Scheduler stopped stepping.")
                    step_scheduler = False
            optimizer.zero_grad()
            its = 0

        if i % (train_cfg["val_freq"] * its_per_step) == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for j, val_batch in enumerate(val_loader):
                    x = batch["input_ids"].to(device, non_blocking=True)
                    logits = model(x[:, :-1])
                    val_loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
                    val_losses.append(val_loss.item())
                    if j == its_per_step * train_cfg["val_batches"]:
                        break
                val_loss = torch.tensor(val_losses).mean().item()
                run.track(val_loss, name="loss", step=i, context={"subset": "val"})
                print(f"val loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": i}, os.path.join(save_dir, "ckpt_best_val.pt"))
                    with open(os.path.join(save_dir, "best_val_loss_step.txt"), "w") as f:
                        f.write(f"loss of {best_loss} achieved on step {i}")
            model.train()


        if stop.is_set():
            print("Received SIGTERM, finishing step and exiting cleanly...")

            break

    print("Saving final weights...")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": i}, os.path.join(save_dir, "ckpt_final.pt"))
    print("storing token_count and tokens per parameter...")
    run["token_count"] = token_count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["tokens_per_parameter"] = token_count / total_params
    run["best_loss"] = best_loss

    run.close()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
