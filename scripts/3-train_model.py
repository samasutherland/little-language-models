import torch
import os
import sys
import time
import uuid
from contextlib import nullcontext

from aim import Run
from lib.run_config import load_run_config
from lib.data_config import build_dataset_and_loader
from lib.model_builder import build_model_from_config
from lib.training_config import (
    build_optimizer,
    build_scheduler,
    build_criterion,
)
from lib.utils import generate_sample


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    amp_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if torch.amp.autocast_mode.is_autocast_available(device) else nullcontext()

    print("Using", device, "device")

    run_config = load_run_config()
    data_config = run_config.load_data()
    train_config = run_config.load_training()

    dataset, loader, vocab_size, pad_id = build_dataset_and_loader(
        data_config,
        train_config.batch_size,
        split=None,
        shuffle=True,
        loop=True,
        num_workers=None,
    )
    _, val_loader, _, _ = build_dataset_and_loader(
        data_config,
        train_config.batch_size,
        split="test",
        shuffle=False,
        loop=False,
    )

    ctx = run_config.get_build_context(vocab_size, pad_id)
    model = build_model_from_config(run_config.model_config_path, ctx=ctx)
    model = model.to(device=device)
    model.train()

    criterion = build_criterion(train_config, ignore_index=pad_id)
    optimizer = build_optimizer(train_config, model)
    its_per_step = train_config.its_per_step()
    scheduler = build_scheduler(train_config, optimizer, its_per_step=its_per_step)

    save_dir = "configs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    best_val_loss = float("inf")

    experiment_name = f"transformer:L{len(model.transformer_stack)}"
    run = Run(experiment=experiment_name)
    run["train_config"] = train_config.model_dump()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["param_count"] = total_params
    try:
        run["data_folder"] = os.environ["RUNPOD_POD_ID"]
    except KeyError:
        run["data_folder"] = uuid.uuid4().hex

    import signal
    import threading
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
        if loss.item() > 10 * torch.median(loss_buffer):
            print("loss huge. skipping...")
            continue

        loss_buffer = push_to_buffer(loss.item())

        if step_scheduler:
            current_lr = scheduler.get_last_lr()[0]
        run.track(loss.item(), name="loss", step=i, context={"subset": "train"})
        if torch.isfinite(loss):
            run.track(
                torch.exp(loss.detach().to(torch.float32).clamp(max=88.72)).item(),
                name="perplexity",
                step=i,
                context={"subset": "train"},
            )
        run.track(current_lr, name="lr", step=i, context={"subset": "train"})

        print(f"Step: {i}, LR: {current_lr}, loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": i},
                os.path.join(save_dir, "ckpt_best.pt"),
            )
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()
            if step_scheduler:
                try:
                    scheduler.step()
                except ValueError:
                    print("Scheduler stopped stepping.")
                    step_scheduler = False
            optimizer.zero_grad()
            its = 0

        if i % (train_config.val_freq * its_per_step) == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for j, val_batch in enumerate(val_loader):
                    x = val_batch["input_ids"].to(device, non_blocking=True)
                    logits = model(x[:, :-1])
                    val_loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        x[:, 1:].reshape(-1),
                    )
                    val_losses.append(val_loss.item())
                    if j >= its_per_step * train_config.val_batches - 1:
                        break
                val_loss = torch.tensor(val_losses).mean().item()
                run.track(val_loss, name="loss", step=i, context={"subset": "val"})
                print(f"val loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": i,
                        },
                        os.path.join(save_dir, "ckpt_best_val.pt"),
                    )
                    with open(os.path.join(save_dir, "best_val_loss_step.txt"), "w") as f:
                        f.write(f"val loss of {best_val_loss} achieved on step {i}")
            model.train()

        if stop.is_set():
            print("Received SIGTERM, finishing step and exiting cleanly...")
            break

    print("Saving final weights...")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": i},
        os.path.join(save_dir, "ckpt_final.pt"),
    )
    run["token_count"] = token_count
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
