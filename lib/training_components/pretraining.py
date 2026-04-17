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

from ..utils import init_datasets_and_models, build_component_from_config, warmup_dataloader
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
        try:
            context, _ = init_datasets_and_models(context)
            evaluation_loop, evaluation_loop_config = build_component_from_config(BenchmarkingLoopFactory,
                                                                                  "configs/training.yaml", context)
            evaluation_loop.descent_steps = self.descent_steps
            # Exclude validation from throughput timing; run configured validation separately.
            evaluation_loop.val_frequency = evaluation_loop.descent_steps + 1
            dataloader_iter = warmup_dataloader(evaluation_loop, context.require("warmup_steps"))
            start = time.perf_counter()
            token_count, *_ = evaluation_loop.run(dataloader_iter=dataloader_iter)
            end = time.perf_counter()
            _ = evaluation_loop.validation_step.step()  # not sure if validation affects the memory usage too much but better to be safe
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
                 min_lr: float,
                 max_lr: float,
                 num_lrs: int,
                 variance_window_size: int,
                 variance_weight: float,
                 ):
        self.sweep_time = sweep_time
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_lrs = num_lrs # 10
        self.variance_window_size = variance_window_size
        self.variance_weight = variance_weight

    def _average_moving_window_variance(self, losses: list[float]) -> float:
        if len(losses) < 2:
            return 0.0
        if self.variance_window_size < 2:
            raise ValueError("variance_window_size must be at least 2.")

        losses_tensor = torch.tensor(losses, dtype=torch.float32)
        max_start = len(losses_tensor) - self.variance_window_size + 1
        if max_start <= 0:
            return torch.var(losses_tensor, unbiased=False).item()

        window_variances = []
        for start_idx in range(max_start):
            window = losses_tensor[start_idx:start_idx + self.variance_window_size]
            window_variances.append(torch.var(window, unbiased=False))
        return torch.stack(window_variances).mean().item()

    @staticmethod
    def _normalize_metric(values: list[float]) -> list[float]:
        if not values:
            return []
        min_value = min(values)
        max_value = max(values)
        value_range = max_value - min_value
        if value_range == 0:
            return [0.0 for _ in values]
        return [(value - min_value) / value_range for value in values]
        
        
    def test_learning_rate(self, context: Context, lr: float):
        lr_descent_steps = int(max(self.sweep_time * context.descent_steps / context.training_time, 1)) ## context.descent_steps / context.training_time is number of steps for 1 minute of training
        # Align validation cadence to the short LR-sweep run length so validation happens on the last step.
        # TrainingLoop validates when i % val_frequency == 0 and i != 0, so val_frequency should be last_step_idx.
        val_frequency = max(lr_descent_steps + 1, 1)
        context = context.fork(
            learning_rate=lr,
            val_frequency=val_frequency,
        )

        # Build with full descent steps so scheduler matches full run, then change descent steps of loop to stop earlier.
        evaluation_loop, evaluation_loop_config = build_component_from_config(BenchmarkingLoopFactory,
                                                                              "configs/training.yaml", context.fork(
                accumulation_steps=max(context.accumulated_batch_size // context.batch_size, 1)))
        # Exclude validation from timing and evaluate once separately using configured validation_batches.
        evaluation_loop.descent_steps = lr_descent_steps
        dataloader_iter = warmup_dataloader(evaluation_loop, context.require("warmup_steps"))
        start = time.perf_counter()
        token_count, loss, val_loss, best_loss, best_val_loss, descent_steps, loss_history = evaluation_loop.run(
            return_loss_history=True,
            dataloader_iter=dataloader_iter,
        )
        end = time.perf_counter()
        val_loss = evaluation_loop.validation_step.step()
        runtime = end - start

        time_per_step = runtime / lr_descent_steps
        total_descent_steps = round((context.training_time * 60) / time_per_step)
        moving_window_variance = self._average_moving_window_variance(loss_history)

        return val_loss, moving_window_variance, total_descent_steps
    
    def run(self, context: Context):
        if self.min_lr <= 0 or self.max_lr <= 0:
            raise ValueError("min_lr and max_lr must be positive for logarithmic spacing.")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr must be less than or equal to max_lr.")

        lrs = torch.logspace(torch.log10(torch.tensor(self.min_lr)),
                            torch.log10(torch.tensor(self.max_lr)),
                            self.num_lrs)
        context, _ = init_datasets_and_models(context, shuffle=False)
        base_state_dict = copy.deepcopy(context.model.state_dict())

        lr_results = {}
        variance_results = {}
        score_results = {}
        lr_test_descent_steps_list = []

        for lr in lrs:
            context.model.load_state_dict(base_state_dict)
            val_loss, moving_window_variance, lr_descent_steps = self.test_learning_rate(context, float(lr))
            lr_test_descent_steps_list.append(lr_descent_steps)
            lr_results[float(lr)] = val_loss
            variance_results[float(lr)] = moving_window_variance

        lr_keys = list(lr_results.keys())
        normalized_val_losses = self._normalize_metric([lr_results[lr] for lr in lr_keys])
        normalized_variances = self._normalize_metric([variance_results[lr] for lr in lr_keys])
        for idx, lr in enumerate(lr_keys):
            score_results[lr] = normalized_val_losses[idx] + self.variance_weight * normalized_variances[idx]

        best_lr = min(score_results, key=score_results.get)
        descent_steps = min(lr_test_descent_steps_list)

        print(f"Best LR is {best_lr}, score={score_results[best_lr]}, val loss={lr_results[best_lr]}, variance={variance_results[best_lr]}")
        print(f"all learning rates:\n {lr_results}")
        print(f"all moving window variances:\n {variance_results}")
        print(f"all combined scores:\n {score_results}")
        
        return context.fork(learning_rate=best_lr, descent_steps=descent_steps)
    
class LearningRateSweepFactory(Factory[LayerSweep]):
    type: Literal["learningratesweep"] = "learningratesweep"

    sweep_time: float
    min_lr: float
    max_lr: float
    num_lrs: int
    variance_window_size: int
    variance_weight: float

    def build(self, ctx: Context) -> LearningRateSweep:

        return LearningRateSweep(sweep_time=self.sweep_time,
                                 min_lr=self.min_lr,
                                 max_lr=self.max_lr,
                                 num_lrs=self.num_lrs,
                                 variance_window_size=self.variance_window_size,
                                 variance_weight=self.variance_weight)

SweepFactory = Annotated[
    Union[LearningRateSweepFactory, LayerSweepFactory],
    Field(discriminator="type"),
]

class Pretrainer:
    def __init__(self,
                 tokens_per_param: int|float,
                 training_time: int|float,
                 warmup_steps: int,
                 layer_sweep: LayerSweep,
                 learning_rate_sweep: LearningRateSweep,):
        
        self.tokens_per_param = tokens_per_param
        self.training_time = training_time
        self.warmup_steps = warmup_steps
        
        self.layer_sweep = layer_sweep
        self.learning_rate_sweep = learning_rate_sweep
        
    def run(self, context: Context):
        context = context.fork(
            tokens_per_param=self.tokens_per_param,
            training_time=self.training_time,
            warmup_steps=self.warmup_steps,
        )
        
        context = self.layer_sweep.run(context)
        context = self.learning_rate_sweep.run(context)
        
        return context
    
class PretrainerFactory(Factory[Pretrainer]):
    type: Literal["pretrainer"] = "pretrainer"
    
    tokens_per_param: int | float
    training_time: int | float
    warmup_steps: int
    layer_sweep: SweepFactory
    learning_rate_sweep: SweepFactory
    
    def build(self, ctx: Context) -> Pretrainer:
        learning_rate_sweep = self.learning_rate_sweep.build(ctx)
        layer_sweep = self.layer_sweep.build(ctx)
        return Pretrainer(tokens_per_param=self.tokens_per_param,
                          training_time=self.training_time,
                          warmup_steps=self.warmup_steps,
                          layer_sweep=layer_sweep,
                          learning_rate_sweep=learning_rate_sweep)