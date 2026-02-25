import gc
import os
import torch
from lib.run_config import load_run_config, load_sweep_spec_for_lr, persist_training_updates
from lib.data_config import build_dataset_and_loader
from lib.model_builder import build_model_from_config
from lib.training_config import build_optimizer, build_scheduler, build_criterion
from lib.utils import get_step_info

print("Loading configs...")
run_config = load_run_config()
data_config = run_config.load_data()
train_config = run_config.load_training()
sweep = load_sweep_spec_for_lr()

print("Getting dataset...")
dataset, loader, vocab_size, pad_id = build_dataset_and_loader(
    data_config,
    train_config.batch_size,
    shuffle=False,
    loop=False,
)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using", device, "device")

train_steps = train_config.total_steps // max(1, int(train_config.training_time))
train_steps = max(train_steps, train_config.warmup_steps * 2)
val_steps = sweep.val_steps_per_trial
print(f"train_steps: {train_steps}, warmup_steps: {train_config.warmup_steps}")

lrs = torch.logspace(sweep.lr_min_exp, sweep.lr_max_exp, sweep.num_lr_trials)
losses = []
last_loss = float("inf")

print("Finding best learning rate...")
for lr in lrs:
    lr_f = float(lr.item())
    torch.manual_seed(42)
    ctx = run_config.get_build_context(vocab_size, pad_id)
    model = build_model_from_config(run_config.model_config_path, ctx=ctx)
    model = model.to(device=device)
    criterion = build_criterion(train_config, ignore_index=pad_id)
    optimizer = build_optimizer(train_config, model, lr=lr_f)
    its_per_step = train_config.its_per_step()
    scheduler = build_scheduler(
        train_config,
        optimizer,
        its_per_step=its_per_step,
        total_training_steps=train_steps * its_per_step,
        warmup_steps=train_config.warmup_steps,
        max_lr=lr_f,
    )
    _, _, loss = get_step_info(
        model,
        device,
        loader,
        criterion,
        optimizer,
        scheduler,
        its_per_step,
        timer_start=0,
        total_steps=train_steps,
        val_steps=val_steps,
    )
    loss_val = loss.item() if hasattr(loss, "item") else float(loss)
    print(f"lr {lr_f} achieved loss {loss_val}")
    losses.append(loss_val)
    if loss_val > last_loss:
        break
    last_loss = loss_val
    del model, optimizer, scheduler, criterion
    gc.collect()

del loader
gc.collect()

losses_t = torch.tensor(losses, dtype=torch.float32)
mask = torch.isfinite(losses_t)
if not torch.any(mask):
    raise RuntimeError("All LR trials produced non-finite losses.")
best_idx = torch.argmin(losses_t[mask]).item()
best_lr = float(lrs[torch.where(mask)[0][best_idx]].item())
print(f"Best learning rate: {best_lr}")
persist_training_updates(run_config.training_config_path, base_lr=best_lr)
