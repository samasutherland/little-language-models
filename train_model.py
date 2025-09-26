import torch, os, sys
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm.auto import tqdm
from aim import Run, Text
import time
import math
import tomllib
import uuid
from lib.helpers import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model_cfg, data_cfg, train_cfg = load_configs()
    dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg)
    model = create_model(model_cfg, vocab_size, device, dataset)
    model.train()

    # Define training params
    criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)


    save_dir = "experiment/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    # optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg["warmup_steps"])
    # decay = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, T_max=train_cfg["total_steps"] - train_cfg["warmup_steps"], **train_cfg["scheduler_kwargs"])
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[train_cfg["warmup_steps"]])

    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=1)
    peak_frac = train_cfg["warmup_steps"] / train_cfg["total_steps"]
    print(f"peak_frac: {peak_frac}")
    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, max_lr=train_cfg["base_lr"], total_steps=train_cfg["total_steps"], pct_start=peak_frac, div_factor=3., final_div_factor=10.)

    experiment_name = "deepseek_transformer" if model_cfg["attention"]["project_kv"] else "dense_transformer"

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
    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(device, non_blocking=True)
        token_count += x[:, 1:].numel()
        logits = model(x[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        current_lr = scheduler.get_last_lr()[0]

        run.track(loss.item(), name="loss", step=i, context={"subset": "train"})
        run.track(current_lr, name="lr", step=i, context={"subset": "train"})


        print(f"Step: {i}, LR: {current_lr}, loss: {loss.item()}")


        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": i}, os.path.join(save_dir, "ckpt_best.pt"))
            with open(os.path.join(save_dir, "best_loss_step.txt"), "w") as f:
                f.write(f"loss of {best_loss} achieved on step {i}")

        if stop.is_set():
            print("Received SIGTERM, finishing step and exiting cleanly...")

            break
    generated_text = generate_sample(model, dataset, device, train_cfg["test_prompt"], n_words=20, max_new_tokens=100,
                                     temperature=1.0, top_k=50,
                                     top_p=0.9)
    print(generated_text)
    run.track(Text(generated_text), name="generated_text")
    run["Final_text_generation"] = generated_text
    run["token_count"] = total_params
    # finally:
    #     try:
    #         start = time.perf_counter()
    #         run.close()
    #         end = time.perf_counter()
    #         print(f"run closed in {end - start:.2f} seconds")
    #     except Exception as e:
    #         print(f"run closure failed")
    #         raise e

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
