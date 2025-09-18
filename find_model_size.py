import torch, os, sys
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import time
import copy
import gc
from lib.helpers import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model
from pathlib import Path
from tomlkit import parse, dumps
from functools import partial
from tqdm import tqdm
import torchinfo
from data.datasets import SimpleStoriesBPEDataset
from datasets import load_dataset

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
SCHEDULER_REGISTRY = {"OneCycleLR": OneCycleLR, "Cosine": CosineAnnealingLR}

def find_token_count(model, device, loader, criterion, optimizer, scheduler):
    step_timer_trigger = 5
    total_steps = 15
    token_count = 0


    for i, batch in enumerate(tqdm(loader, leave=True, desc="Finding Token Count")):
        x = batch["input_ids"].to(device=device, non_blocking=True)

        if i == step_timer_trigger:
            token_count = 0
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        token_count += torch.prod(torch.tensor(x.shape)).item()
        logits = model(x[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i == total_steps - 1:
            break
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("finished finding token count")
    total_tokens = token_count * 1800 / (end_time - start_time) # number of tokens in 30 min
    number_of_steps = int((total_steps - step_timer_trigger) * 1800 / (end_time - start_time))
    return total_tokens, number_of_steps


# def find_max_batch_size(model, dataset, device, criterion, optimizer, scheduler, data_cfg, train_cfg, collate, starting_size=1):
#     def check_fits(batch_size, test_steps=3):
#         try:
#             cfg = copy.deepcopy(train_cfg)
#             cfg["batch_size"] = batch_size
#             loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
#                                 num_workers=0, persistent_workers=False)
#
#             for i, batch in enumerate(loader):
#                 x = batch["input_ids"].to(device, non_blocking=True)
#                 logits = model(x[:, :-1])
#                 loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#                 if i > test_steps:
#                     break
#         except RuntimeError as e:
#             if device == "cuda":
#                 torch.cuda.empty_cache()
#             gc.collect()
#             return False
#         return True
#
#     min_batch = 1
#     batch_size = starting_size
#     while check_fits(batch_size):
#         gc.collect()
#         min_batch = batch_size
#         batch_size = 2 * min_batch
#
#     max_batch = batch_size
#
#     while max_batch - min_batch > 1:
#         batch_size = min_batch + (max_batch - min_batch)//2
#         if check_fits(batch_size):
#             gc.collect()
#             min_batch = batch_size
#         else:
#             gc.collect()
#             max_batch = batch_size
#
#     return min_batch

def find_max_batch_size(model, dataset, available_memory, starting_size=1, safety_factor=1000000):
    batch_size = starting_size
    model_statistics = torchinfo.summary(model, input_size=[(1, dataset.max_length,)], dtypes=[torch.long])

    # total_param_bytes doubled to include .grad buffers also.
    fixed_size = 2*model_statistics.total_param_bytes + (2 * model_statistics.trainable_params * 2) # optimizer parameters -> 2 parameters per trainable parameter, 2bytes per parameter(bfloat16)
    variable_size = model_statistics.total_input + model_statistics.total_output_bytes
    print(f"fixed bytes: {fixed_size}")
    print(f"bytes per batch: {variable_size}")

    total_size = fixed_size + (variable_size * batch_size)
    if total_size > (available_memory - safety_factor):
        raise Exception("The starting batch size is too big")

    while total_size < (available_memory - safety_factor):
        batch_size *= 2
        total_size = fixed_size + (variable_size * batch_size)

    return batch_size // 2



def adjust_model_parameters(target_parameter_count, model_cfg, vocab_size, device, dataset, starting_num_layers=1):
    def compute_total_params(num_layers):
        model_cfg["global"]["num_layers"] = num_layers
        model_cfg["global"]["embedding_dim"] = num_layers * 128
        model_cfg["global"]["feedforward_dim"] = num_layers * 128 * 4
        model_cfg["attention"]["n_heads"] = num_layers

        model = create_model(model_cfg, vocab_size, device, dataset)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return total_params

    min_layers = 1
    num_layers = starting_num_layers
    while compute_total_params(num_layers) < target_parameter_count:
        gc.collect()
        min_layers = num_layers
        num_layers = num_layers * 2

    max_layers = num_layers

    while max_layers - min_layers > 1:
        num_layers = min_layers + (max_layers - min_layers) // 2
        if compute_total_params(num_layers) < target_parameter_count:
            gc.collect()
            min_layers = num_layers
        else:
            max_layers = num_layers

    min_diff = target_parameter_count - compute_total_params(min_layers)
    max_diff = compute_total_params(max_layers) - target_parameter_count
    gc.collect()

    if min_diff > max_diff:
        model_cfg["global"]["num_layers"] = max_layers
        model_cfg["global"]["embedding_dim"] = max_layers * 128
        model_cfg["global"]["feedforward_dim"] = max_layers * 128 * 4
        model_cfg["attention"]["n_heads"] = max_layers
    else:
        model_cfg["global"]["num_layers"] = min_layers
        model_cfg["global"]["embedding_dim"] = min_layers * 128
        model_cfg["global"]["feedforward_dim"] = min_layers * 128 * 4
        model_cfg["attention"]["n_heads"] = min_layers


def main():
    print("loading configs...")
    model_cfg_path = Path("experiment/model.toml")
    data_cfg_path = Path("experiment/data.toml")
    train_cfg_path = Path("experiment/training.toml")

    model_cfg = parse(model_cfg_path.read_text(encoding="utf-8"))
    data_cfg = parse(data_cfg_path.read_text(encoding="utf-8"))
    train_cfg = parse(train_cfg_path.read_text(encoding="utf-8"))

    print("getting dataset")
    data = load_dataset(data_cfg["dataset"])
    dataset = SimpleStoriesBPEDataset(data[data_cfg["split"]], max_length=data_cfg["max_length"])

    collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

    torch.set_default_dtype(torch.bfloat16)
    vocab_size = max(
        max(dataset.tok.vocab.keys(), default=-1),
        max(getattr(dataset.tok, "merge_dict", {}).values(), default=-1),
        dataset.pad_id,
        dataset.end_id,
    ) + 1
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    free_mem, total_mem = torch.cuda.memory.mem_get_info()

    print(f"Using {device} device")
    for i in range(10):
        print(f"it {i} num_layers: {model_cfg['global']['num_layers']}")
        print("creating model...")
        model = create_model(model_cfg, vocab_size, device, dataset)
        model.train()
        print("model created.")

        # Define training params
        criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)
        optimizer = OPTIMIZER_REGISTRY[train_cfg["optimizer"]](model.parameters(), lr=train_cfg["base_lr"])

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=train_cfg["warmup_steps"])
        decay = SCHEDULER_REGISTRY[train_cfg["scheduler"]](optimizer, T_max=train_cfg["total_steps"] - train_cfg["warmup_steps"], **train_cfg["scheduler_kwargs"])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[train_cfg["warmup_steps"]])

        print("finding max batch size...")
        batch_size = find_max_batch_size(model, dataset, free_mem, starting_size=16)
        print(f"Max batch size: {batch_size}")
        train_cfg["batch_size"] = batch_size
        loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
                            num_workers=4, persistent_workers=False)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Computing throughput and effective number of tokens...")
        total_tokens, total_steps = find_token_count(model, device, loader, criterion, optimizer, scheduler)

        train_cfg["total_steps"] = total_steps

        print(f"Total tokens: {total_tokens}, total params: {total_params}, ratio: {total_tokens / total_params} (target 20)")

        if abs((total_tokens / total_params) - 20) < 2: # target is 20 tokens per parameter
            break

        target_parameter_count = total_tokens // 20

        adjust_model_parameters(target_parameter_count, model_cfg, vocab_size, device, dataset, starting_num_layers=model_cfg["global"]["num_layers"])

        del model, loader, dataset, optimizer, criterion, scheduler
        torch.cuda.empty_cache()
        gc.collect()

        model_cfg_path.write_text(dumps(model_cfg), encoding="utf-8")
        data_cfg_path.write_text(dumps(data_cfg), encoding="utf-8")
        train_cfg_path.write_text(dumps(train_cfg), encoding="utf-8")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
