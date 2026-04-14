from sympy import divisors
import math

from lib.utils import init_train_device, init_datasets, init_datasets_and_models, init_runtime_contexts, \
    fibonacci_search
from lib.component_builder import build_component_from_config

from lib import Context
import copy


from lib.model_components import LanguageModelFactory
from lib.data_components import DataLoaderFactory
from lib.training_components.loops import LoopFactory, BenchmarkingLoopFactory, TrainingLoopFactory

import torch
from contextlib import nullcontext
from pathlib import Path
import yaml
import time
torch.manual_seed(42)

def test_memory_fits(context: Context):
    context = context.fork(descent_steps=10)
    try: 
        context, _ = init_datasets_and_models(context)
        evaluation_loop, evaluation_loop_config = build_component_from_config(BenchmarkingLoopFactory,
                                                                              "configs/training.yaml", context)
        start = time.perf_counter()
        token_count, *_ = evaluation_loop.run()
        end = time.perf_counter()
        runtime = end - start
        
        tokens_per_second = token_count / runtime
        total_tokens = tokens_per_second * context.training_time * 60
        total_parameters = sum(p.numel() for p in context.require("model").parameters() if p.requires_grad)
        tokens_per_parameter = total_tokens / total_parameters

        time_per_step = runtime / context.descent_steps
        total_descent_steps = round((context.training_time * 60) / time_per_step)
        return True, tokens_per_parameter, total_descent_steps
        
    except RuntimeError as e:
        print(e)
        return False, 0, 0

def find_batch_size(context):
    
    token_throughputs = {}

    accumulated_batch_size = context.require("accumulated_batch_size")
    batch_sizes = list(divisors(accumulated_batch_size))[::-1]
    batch_sizes.insert(0, accumulated_batch_size * 2)

    any_success = False
    for i, batch_size in enumerate(batch_sizes):
        success, tokens_per_parameter, total_descent_steps = test_memory_fits(context.fork(batch_size=batch_size, accumulation_steps=max(context.accumulated_batch_size//batch_size, 1)))
        if success:
            # print(f"Batch size {batch_size} passed")
            any_success = True
            break
            
    if not any_success:
        # print("No batch sizes successful.")
        return 0, 0, 0
    final_batch_size = max(batch_size // 2, 1)
    _, tokens_per_parameter, total_descent_steps = test_memory_fits(context.fork(batch_size=final_batch_size, accumulation_steps=max(context.accumulated_batch_size//final_batch_size, 1)))
    return final_batch_size, tokens_per_parameter, total_descent_steps

def test_learning_rate(context, lr):
    lr_descent_steps = max(context.descent_steps//context.training_time, 1)
    context = context.fork(learning_rate=lr, descent_steps=lr_descent_steps)
    
    # To fix: scheduler has peak_frac at 0.001, which will be much earlier here because the descent_steps is much smaller.
    # However, that may be ok as it will just basically do max rate from the start, which may be what we want.
    evaluation_loop, evaluation_loop_config = build_component_from_config(BenchmarkingLoopFactory,
                                                                          "configs/training.yaml", context.fork(accumulation_steps=max(context.accumulated_batch_size//context.batch_size, 1)))
    start = time.perf_counter()
    token_count, loss, val_loss, best_loss, best_val_loss, descent_steps = evaluation_loop.run()
    end = time.perf_counter()
    runtime = end - start

    time_per_step = runtime / context.descent_steps
    total_descent_steps = round((context.training_time * 60) / time_per_step)
    
    return best_val_loss, total_descent_steps


def main():
    device, autocast_context = init_train_device()
    context = Context(autocast_ctx=autocast_context, device=device)
    
    runtime_context, _ = init_runtime_contexts()
    context.merge(runtime_context)
    
    pretraining_path = Path("configs/pretraining.yaml")
    with pretraining_path.open("r") as f:
        pretraining_dict = yaml.safe_load(f)
    context.merge(Context(**pretraining_dict))
    
    # Find max num layers that still processes at least target tokens per parameter
    num_layers = context.require("num_layers_lower_bound")
    
    tokens_per_parameter_dict = {}
    batch_size_dict = {}
    descent_steps_dict = {}
    
    # Start by sequentially doubling the num_layers
    counter = 1
    while True:
        # print(f"testing {num_layers} layers")
        batch_size, tokens_per_parameter, total_descent_steps = find_batch_size(context.fork(num_layers=num_layers))
        tokens_per_parameter_dict[num_layers] = tokens_per_parameter
        batch_size_dict[num_layers] = batch_size
        descent_steps_dict[num_layers] = total_descent_steps
        print(f"{num_layers} layer model uses batch size {batch_size} and gets {tokens_per_parameter} tokens per parameter")
        if tokens_per_parameter < context.tokens_per_param:
            break
        num_layers *= 2
        counter += 1
        
    if counter == 1:
        raise ValueError("Smallest model can't reach required token throughput. Decrease smallest model size.")
    # Now a binary search to find the best number of layers
    upper_bound = num_layers
    lower_bound = num_layers // 2
    
    while upper_bound - lower_bound > 1:
        num_layers = lower_bound + (upper_bound - lower_bound) // 2
        batch_size, tokens_per_parameter, total_descent_steps = find_batch_size(context.fork(num_layers=num_layers))
        tokens_per_parameter_dict[num_layers] = tokens_per_parameter
        batch_size_dict[num_layers] = batch_size
        descent_steps_dict[num_layers] = total_descent_steps
        print(
            f"{num_layers} layer model uses batch size {batch_size} and gets {tokens_per_parameter} tokens per parameter")
        
        if tokens_per_parameter >= context.tokens_per_param:
            lower_bound = num_layers
        else:
            upper_bound = num_layers
    # old method, choose closest value instead of maximum size above target
    # targ_dists = {layer_count: abs(context.tokens_per_param - tokens_per_parameter_dict[layer_count]) for layer_count in tokens_per_parameter_dict.keys()}
    # num_layers = min(targ_dists, key=targ_dists.get) 
    
    num_layers = lower_bound
    batch_size = batch_size_dict[num_layers]
    descent_steps = descent_steps_dict[num_layers]
    print(f"Final model has {num_layers} layers, uses batch size {batch_size} and gets {tokens_per_parameter_dict[num_layers]} tokens per parameter, running for {descent_steps} steps")
    
    context.merge({"num_layers": num_layers, "batch_size": batch_size, "accumulation_steps": max(context.accumulated_batch_size // batch_size, 1), "descent_steps": descent_steps})
    
    # ------------------- Learning rate tuning --------------------- # 

    min_lr_exp = -3
    max_lr_exp = 0
    lrs = torch.logspace(min_lr_exp, max_lr_exp, 10)
    context, _ = init_datasets_and_models(context, shuffle=False)
    base_state_dict = copy.deepcopy(context.model.state_dict())
    
    lr_results = {}
    lr_test_descent_steps_list = []
    
    for lr in lrs:
        context.model.load_state_dict(base_state_dict)
        val_loss, lr_descent_steps = test_learning_rate(context, float(lr))
        lr_test_descent_steps_list.append(lr_descent_steps)
        lr_results[float(lr)] = val_loss
        
    best_lr = min(lr_results, key=lr_results.get)
    descent_steps = min(lr_test_descent_steps_list)
    
    print(f"Best LR is {best_lr}, achieved val loss of {lr_results[best_lr]}")
    print(f"all learning rates:\n {lr_results}")
    
    context_path = Path("configs/context.yaml")
    with context_path.open("r") as f:
        run_context_dict = yaml.safe_load(f)
    run_context_dict["num_layers"] = num_layers
    run_context_dict["batch_size"] = batch_size
    run_context_dict["descent_steps"] = descent_steps
    run_context_dict["learning_rate"] = best_lr
    with context_path.open("w") as f:
        yaml.safe_dump(run_context_dict, f)
        
        
if __name__ == "__main__":
    main()
