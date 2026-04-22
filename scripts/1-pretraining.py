from sympy import divisors
import math

from lib.training_components.pretraining import PretrainerFactory
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


def main():
    device, autocast_context = init_train_device()
    context = Context(autocast_ctx=autocast_context, device=device)
    
    runtime_context, _ = init_runtime_contexts()
    context.merge(runtime_context)
    torch.manual_seed(context.require("seed"))

    pretrainer, pretraining_context = build_component_from_config(PretrainerFactory, "configs/pretraining.yaml", context)
    context = pretrainer.run(context)
    
    context_path = Path("configs/context.yaml")
    with context_path.open("r") as f:
        run_context_dict = yaml.safe_load(f)
    run_context_dict["num_layers"] = context.num_layers
    run_context_dict["batch_size"] = context.batch_size
    run_context_dict["accumulated_batch_size"] = context.accumulated_batch_size
    run_context_dict["descent_steps"] = context.descent_steps
    run_context_dict["learning_rate"] = context.learning_rate
    run_context_dict["vocab_size"] = context.require("vocab_size")
    with context_path.open("w") as f:
        yaml.safe_dump(run_context_dict, f)
        
        
if __name__ == "__main__":
    main()
