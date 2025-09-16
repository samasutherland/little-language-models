import torch, os, sys
from datasets import load_dataset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from functools import partial
from data.datasets import SimpleStoriesBPEDataset
from lib.models import MODEL_REGISTRY
from aim import Run
import math
import tomllib
from lib.helpers import generate_sample

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR}


def pad_collate_fn(batch, pad_id):
    x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    m = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": m}

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Import configs
    with open("experiment/model.toml", "rb") as f:
        model_cfg = tomllib.load(f)
    with open("experiment/data.toml", "rb") as f:
        data_cfg = tomllib.load(f)
    with open("experiment/training.toml", "rb") as f:
        train_cfg = tomllib.load(f)

    # Define data loader
    data = load_dataset(data_cfg["dataset"])
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=4, persistent_workers=True)

    torch.set_default_dtype(torch.bfloat16)
    vocab_size = max(
        max(dataset.tok.vocab.keys(), default=-1),
        max(getattr(dataset.tok, "merge_dict", {}).values(), default=-1),
        dataset.pad_id,
        dataset.end_id,
    ) + 1

    # Define model
    model_cls = MODEL_REGISTRY[model_cfg["model"]]

    model_cfg["global"] = model_cfg["global"] | {"vocab_size": vocab_size}

    model = model_cls(model_cfg).to(dtype=torch.bfloat16, device=device)

    model.embedding.weight.data = model.embedding.weight.data.to(torch.bfloat16)
    model.embedding.padding_idx = dataset.pad_id

    # Define training params

    criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)
    optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])

    epochs = train_cfg["epochs"]
    total_steps = train_cfg["total_steps"]
    save_dir = "experiment/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0
    loss_steps, loss_values = [], []
    prompt_text = train_cfg["test_prompt"]

    start_epoch = 0
    scheduler = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, total_steps=total_steps, **train_cfg["scheduler_kwargs"])

    experiment_name = "deepseek_transformer" if model_cfg["attention"]["project_kv"] else "dense_transformer"

    run = Run(experiment_name = experiment_name,
              run_hash = os.environ["RUNPOD_POD_ID"])
    run["model_cfg"] = model_cfg
    run["data_cfg"] = data_cfg
    run["train_cfg"] = train_cfg

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["param_count"] = total_params

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", total=total_steps, file=sys.stdout, miniters=10, disable=True)
        epoch_loss, n_batches = 0.0, 0
        for i, batch in enumerate(pbar):
            x = batch["input_ids"].to(device, non_blocking=True)
            logits = model(x[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            loss.backward()
            loss_steps.append(global_step)
            loss_values.append(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")

            lrs = scheduler.get_last_lr()
            mult_now = lrs[0] / scheduler.max_lrs[0]

            run.track(loss.item(), name="loss", step=i, context={"subset": "train"})
            run.track(mult_now, name="lr_multiplier", step=i, context={"subset": "train"})

            print(f"LR: {mult_now}, loss: {loss.item()}")


            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": global_step, "epoch": epoch}, os.path.join(save_dir, "ckpt_best.pt"))
                with open(os.path.join(save_dir, "best_loss_step.txt"), "w") as f:
                    f.write(f"loss of {best_loss} achieved on step {i}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
