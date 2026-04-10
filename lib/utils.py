import torch
from contextlib import nullcontext
from pathlib import Path
import yaml

from lib.component_builder import build_component_from_config

from lib.data_components import DataLoaderFactory  
from lib.model_components import LanguageModelFactory
from lib.base_classes import Context

import functools

def init_train_device():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    amp_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if torch.amp.autocast_mode.is_autocast_available(
        device) else nullcontext()
    
    return device, amp_ctx

def init_datasets(context):
    train_dataloader = build_component_from_config(DataLoaderFactory, "../configs/data.yaml",
                                                   context.fork(split="train"))
    val_dataloader = build_component_from_config(DataLoaderFactory, "../configs/data.yaml", context.fork(split="test"))
    return train_dataloader, val_dataloader

def init_datasets_and_models(context, shuffle=True):
    (train_dataloader, data_config), (val_dataloader, data_config) = init_datasets(context.fork(shuffle=shuffle))
    context.merge({"train_dataloader": train_dataloader, "val_dataloader": val_dataloader})
    model, model_config = build_component_from_config(LanguageModelFactory, "../configs/model.yaml", context)
    device = context.require("device")
    model = model.to(device)
    context.merge({"model": model})
    return context, {"data": data_config, "model": model_config}

def init_runtime_contexts():
    context_path = Path("../configs/context.yaml")
    with context_path.open("r") as f:
        run_context_dict = yaml.safe_load(f)
    context = Context(**run_context_dict)

    server_path = Path("../configs/server.yaml")
    with server_path.open("r") as f:
        server_dict = yaml.safe_load(f)
    context.merge(Context(**server_dict))
    return context, {"run_context": run_context_dict, "server": server_dict}


def fibonacci_search(func, func_args=(), func_kwargs=None, lower_bound=1, upper_bound=32):
    if func_kwargs is None:
        func_kwargs = {}
    @functools.lru_cache
    def func_(probe):
        print(f"evaluating {probe}")
        return func(probe, *func_args, **func_kwargs)


    fib_nums = [1,1]
    while fib_nums[-1] <= (upper_bound - lower_bound):
        fib_nums.append(fib_nums[-1] + fib_nums[-2])

    fib_nums = fib_nums[::-1][1:]
    fib_index = 0

    # Initial pass uses two evaluations
    probe_upper = lower_bound + fib_nums[fib_index] - 1
    probe_lower = lower_bound + fib_nums[fib_index + 1] - 1

    result_upper = func_(probe_upper)
    result_lower = func_(probe_lower)


    if result_upper < result_lower:
        lower_bound = probe_lower + 1
        probe_lower = probe_upper
        result_lower = result_upper
        flag = "UPPER"
    else:
        upper_bound = probe_upper - 1
        probe_upper = probe_lower
        result_upper = result_lower
        flag = "LOWER"

    fib_index += 1

    while upper_bound - lower_bound > 1:
        if flag == "UPPER":
            probe_upper = lower_bound + fib_nums[fib_index] - 1
            result_upper = func_(probe_upper)

        elif flag == "LOWER":
            probe_lower = lower_bound + fib_nums[fib_index + 1] - 1
            result_lower = func_(probe_lower)

        if result_upper < result_lower:
            lower_bound = probe_lower + 1
            probe_lower = probe_upper
            result_lower = result_upper
            flag = "UPPER"
        else:
            upper_bound = probe_upper - 1
            probe_upper = probe_lower
            result_upper = result_lower
            flag = "LOWER"

        fib_index += 1

    result_upper = func_(upper_bound)
    result_lower = func_(lower_bound)
    if result_upper < result_lower:
        return upper_bound
    else:
        return lower_bound
# 
# 
