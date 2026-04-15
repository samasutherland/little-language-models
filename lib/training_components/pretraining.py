import warnings
from typing import Literal, Annotated, Union, Any, Dict
from pydantic import Field, ConfigDict
from torch.utils.data import DataLoader

from tqdm import tqdm
from lib import Context, Factory


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pathlib import Path
import os
import copy

from aim import Run
import yaml

from lib.training_components import OptimizerFactory
from lib.training_components.steps import EvaluationStep, GradientStep, ValidationStep, StepFactory
from lib.training_components.loops import BenchmarkingLoopFactory
import time

from ..utils import init_datasets_and_models, build_component_from_config
from sympy import divisors


class LayerSweep:
    def __init__(self,
                 descent_steps: int,
                 lower_bound: int,
                 method: str):
        self.descent_steps = descent_steps
        self.lower_bound = lower_bound
        self.method = method
        if self.method not in ["closest", "first_above"]:
            raise ValueError("method must be either closest or first_above")

    def test_memory_fits(self, context: Context):
        context = context.fork(descent_steps=self.descent_steps)
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

    def find_batch_size(self, context: Context):
        accumulated_batch_size = context.require("accumulated_batch_size")
        batch_sizes = list(divisors(accumulated_batch_size))[::-1]
        batch_sizes.insert(0, accumulated_batch_size * 2)
    
        any_success = False
        for i, batch_size in enumerate(batch_sizes):
            success, tokens_per_parameter, total_descent_steps = self.test_memory_fits(context.fork(batch_size=batch_size,
                                                                                               accumulation_steps=max(
                                                                                                   context.accumulated_batch_size // batch_size,
                                                                                                   1)))
            if success:
                # print(f"Batch size {batch_size} passed")
                any_success = True
                break
    
        if not any_success:
            # print("No batch sizes successful.")
            return 0, 0, 0
        final_batch_size = max(batch_size // 2, 1)
        _, tokens_per_parameter, total_descent_steps = self.test_memory_fits(context.fork(batch_size=final_batch_size,
                                                                                     accumulation_steps=max(
                                                                                         context.accumulated_batch_size // final_batch_size,
                                                                                         1)))
        return final_batch_size, tokens_per_parameter, total_descent_steps
    
    def run(self, 
            context: Context,
            ):
        tokens_per_parameter_dict = {}
        batch_size_dict = {}
        descent_steps_dict = {}
        
        num_layers = self.lower_bound
        counter = 1
        while True:
            # print(f"testing {num_layers} layers")
            batch_size, tokens_per_parameter, total_descent_steps = self.find_batch_size(context.fork(num_layers=num_layers))
            tokens_per_parameter_dict[num_layers] = tokens_per_parameter
            batch_size_dict[num_layers] = batch_size
            descent_steps_dict[num_layers] = total_descent_steps
            print(
                f"{num_layers} layer model uses batch size {batch_size} and gets {tokens_per_parameter} tokens per parameter")
            if tokens_per_parameter < context.tokens_per_param:
                break
            num_layers *= 2
            counter += 1

        if counter == 1:
            raise ValueError("Smallest model can't reach required token throughput. Decrease smallest model size.")
        # Now a binary search to find the best number of layers
        upper_bound = num_layers
        lower_bound = num_layers // 2
        
        # Now perform binary search to find specific layer count
        while upper_bound - lower_bound > 1:
            num_layers = lower_bound + (upper_bound - lower_bound) // 2
            batch_size, tokens_per_parameter, total_descent_steps = self.find_batch_size(context.fork(num_layers=num_layers))
            tokens_per_parameter_dict[num_layers] = tokens_per_parameter
            batch_size_dict[num_layers] = batch_size
            descent_steps_dict[num_layers] = total_descent_steps
            print(
                f"{num_layers} layer model uses batch size {batch_size} and gets {tokens_per_parameter} tokens per parameter")

            if tokens_per_parameter >= context.tokens_per_param:
                lower_bound = num_layers
            else:
                upper_bound = num_layers

        if self.method == "closest":
            targ_dists = {layer_count: abs(context.tokens_per_param - tokens_per_parameter_dict[layer_count]) for layer_count in tokens_per_parameter_dict.keys()}
            num_layers = min(targ_dists, key=targ_dists.get) 
        elif self.method == "first_above":
            num_layers = lower_bound
        else:
            raise ValueError("method must be either closest or first_above")

        batch_size = batch_size_dict[num_layers]
        descent_steps = descent_steps_dict[num_layers]

        print(
            f"Final model has {num_layers} layers, uses batch size {batch_size} and gets {tokens_per_parameter_dict[num_layers]} tokens per parameter, running for {descent_steps} steps")

        return context.fork(num_layers=num_layers, batch_size=batch_size,
                       accumulation_steps=max(context.accumulated_batch_size // batch_size, 1),
                       descent_steps=descent_steps)


class LayerSweepFactory(Factory[LayerSweep]):
    type: Literal["layersweep"] = "layersweep"

    descent_steps: int
    lower_bound: int
    method: str

    def build(self, ctx: Context) -> LayerSweep:

        return LayerSweep(descent_steps=self.descent_steps,
                          lower_bound=self.lower_bound,
                          method=self.method,)
    
class LearningRateSweep:
    def __init__(self,
                 sweep_time: float,
                 min_lr_exp: float,
                 max_lr_exp: float,
                 num_lrs: int,
                 ):
        self.sweep_time = sweep_time
        self.min_lr_exp = min_lr_exp # -3
        self.max_lr_exp = max_lr_exp # 0
        self.num_lrs = num_lrs # 10
        
        
    def test_learning_rate(self, context: Context, lr: float):
        lr_descent_steps = int(max(self.sweep_time * context.descent_steps / context.training_time, 1)) ## context.descent_steps / context.training_time is number of steps for 1 minute of training
        context = context.fork(learning_rate=lr, descent_steps=lr_descent_steps)

        # To fix: scheduler has peak_frac at 0.001, which will be much earlier here because the descent_steps is much smaller.
        # However, that may be ok as it will just basically do max rate from the start, which may be what we want.
        evaluation_loop, evaluation_loop_config = build_component_from_config(BenchmarkingLoopFactory,
                                                                              "configs/training.yaml", context.fork(
                accumulation_steps=max(context.accumulated_batch_size // context.batch_size, 1)))
        start = time.perf_counter()
        token_count, loss, val_loss, best_loss, best_val_loss, descent_steps = evaluation_loop.run()
        end = time.perf_counter()
        runtime = end - start

        time_per_step = runtime / context.descent_steps
        total_descent_steps = round((context.training_time * 60) / time_per_step)

        return best_val_loss, total_descent_steps
    
    def run(self, context: Context):
        lrs = torch.logspace(self.min_lr_exp, self.max_lr_exp, self.num_lrs)
        context, _ = init_datasets_and_models(context, shuffle=False)
        base_state_dict = copy.deepcopy(context.model.state_dict())

        lr_results = {}
        lr_test_descent_steps_list = []

        for lr in lrs:
            context.model.load_state_dict(base_state_dict)
            val_loss, lr_descent_steps = self.test_learning_rate(context, float(lr))
            lr_test_descent_steps_list.append(lr_descent_steps)
            lr_results[float(lr)] = val_loss

        best_lr = min(lr_results, key=lr_results.get)
        descent_steps = min(lr_test_descent_steps_list)

        print(f"Best LR is {best_lr}, achieved val loss of {lr_results[best_lr]}")
        print(f"all learning rates:\n {lr_results}")
        
        return context.fork(learning_rate=best_lr, descent_steps=descent_steps)
    
class LearningRateSweepFactory(Factory[LayerSweep]):
    type: Literal["learningratesweep"] = "learningratesweep"

    sweep_time: float
    min_lr_exp: float
    max_lr_exp: float
    num_lrs: int

    def build(self, ctx: Context) -> LearningRateSweep:

        return LearningRateSweep(sweep_time=self.sweep_time,
                                 min_lr_exp=self.min_lr_exp,
                                 max_lr_exp=self.max_lr_exp,
                                 num_lrs=self.num_lrs)

SweepFactory = Annotated[
    Union[LearningRateSweepFactory, LayerSweepFactory],
    Field(discriminator="type"),
]

class Pretrainer:
    def __init__(self,
                 tokens_per_param: int|float,
                 training_time: int|float,
                 layer_sweep: LayerSweep,
                 learning_rate_sweep: LearningRateSweep,):
        
        self.tokens_per_param = tokens_per_param
        self.training_time = training_time
        
        self.layer_sweep = layer_sweep
        self.learning_rate_sweep = learning_rate_sweep
        
    def run(self, context: Context):
        context = context.fork(tokens_per_param=self.tokens_per_param, training_time=self.training_time)
        
        context = self.layer_sweep.run(context)
        context = self.learning_rate_sweep.run(context)
        
        return context
    
class PretrainerFactory(Factory[Pretrainer]):
    type: Literal["pretrainer"] = "pretrainer"
    
    tokens_per_param: int | float
    training_time: int | float
    layer_sweep: SweepFactory
    learning_rate_sweep: SweepFactory
    
    def build(self, ctx: Context) -> Pretrainer:
        learning_rate_sweep = self.learning_rate_sweep.build(ctx)
        layer_sweep = self.layer_sweep.build(ctx)
        return Pretrainer(tokens_per_param=self.tokens_per_param,
                          training_time=self.training_time,
                          layer_sweep=layer_sweep,
                          learning_rate_sweep=learning_rate_sweep)