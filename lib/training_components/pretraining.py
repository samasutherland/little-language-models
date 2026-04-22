from typing import Any, Literal

import optuna
import time
from optuna.exceptions import TrialPruned
from sympy import divisors

from lib import Context, Factory
from lib.training_components.loops import BenchmarkingLoopFactory
from ..utils import init_datasets_and_models, build_component_from_config, warmup_dataloader


class BatchSizeFinder:
    def __init__(self, descent_steps: int):
        self.descent_steps = descent_steps

    def test_memory_fits(self, context: Context) -> tuple[bool, int]:
        try:
            context, _ = init_datasets_and_models(context)
            evaluation_loop, _ = build_component_from_config(BenchmarkingLoopFactory, "configs/training.yaml", context)
            evaluation_loop.descent_steps = self.descent_steps
            evaluation_loop.val_frequency = evaluation_loop.descent_steps + 1
            dataloader_iter = warmup_dataloader(evaluation_loop, context.require("warmup_steps"))
            start = time.perf_counter()
            _ = evaluation_loop.run(dataloader_iter=dataloader_iter)
            end = time.perf_counter()
            _ = evaluation_loop.validation_step.step()
            runtime = end - start
            time_per_step = runtime / self.descent_steps
            total_descent_steps = round((context.training_time * 60) / time_per_step)
            return True, total_descent_steps
        except RuntimeError as e:
            print(e)
            return False, 0

    def find_batch_size(self, context: Context) -> tuple[int, int]:
        accumulated_batch_size = context.require("accumulated_batch_size")
        batch_sizes = list(divisors(accumulated_batch_size))[::-1]

        for batch_size in batch_sizes:
            success, total_descent_steps = self.test_memory_fits(
                context.fork(
                    batch_size=batch_size,
                    accumulation_steps=max(context.accumulated_batch_size // batch_size, 1),
                )
            )
            if success:
                return max(batch_size, 1), total_descent_steps
        return 0, 0


class BatchSizeFinderFactory(Factory[BatchSizeFinder]):
    type: Literal["batchsizefinder"] = "batchsizefinder"
    descent_steps: int

    def build(self, ctx: Context) -> BatchSizeFinder:
        return BatchSizeFinder(descent_steps=self.descent_steps)


class OptunaSearch:
    def __init__(
        self,
        n_trials: int,
        sweep_time: float,
        min_lr: float,
        max_lr: float,
        accumulated_batch_sizes: list[int],
        min_num_layers: int,
        max_num_layers: int,
        vocab_sizes: list[int],
        startup_trials: int,
    ):
        self.n_trials = n_trials
        self.sweep_time = sweep_time
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.accumulated_batch_sizes = accumulated_batch_sizes
        self.min_num_layers = min_num_layers
        self.max_num_layers = max_num_layers
        self.vocab_sizes = vocab_sizes
        self.startup_trials = startup_trials

        if self.min_lr <= 0 or self.max_lr <= 0:
            raise ValueError("min_lr and max_lr must be positive.")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr must be <= max_lr.")
        if self.min_num_layers > self.max_num_layers:
            raise ValueError("min_num_layers must be <= max_num_layers.")
        if len(self.accumulated_batch_sizes) == 0:
            raise ValueError("accumulated_batch_sizes cannot be empty.")

    def _trial_context(self, trial: optuna.Trial, context: Context) -> Context:
        learning_rate = trial.suggest_float("learning_rate", self.min_lr, self.max_lr, log=True)
        accumulated_batch_size = trial.suggest_categorical("accumulated_batch_size", self.accumulated_batch_sizes)
        num_layers = trial.suggest_int("num_layers", self.min_num_layers, self.max_num_layers)
        updates: dict[str, Any] = {
            "learning_rate": learning_rate,
            "accumulated_batch_size": accumulated_batch_size,
            "num_layers": num_layers,
        }
        if self.vocab_sizes:
            updates["vocab_size"] = trial.suggest_categorical("vocab_size", self.vocab_sizes)
        return context.fork(**updates)

    def _evaluate_trial(self, context: Context) -> float:
        context, _ = init_datasets_and_models(context, shuffle=False)
        evaluation_loop, _ = build_component_from_config(
            BenchmarkingLoopFactory,
            "configs/training.yaml",
            context.fork(accumulation_steps=max(context.accumulated_batch_size // context.batch_size, 1)),
        )
        trial_descent_steps = max(int(self.sweep_time * context.descent_steps / context.training_time), 1)
        evaluation_loop.descent_steps = trial_descent_steps
        dataloader_iter = warmup_dataloader(evaluation_loop, context.require("warmup_steps"))
        _ = evaluation_loop.run(dataloader_iter=dataloader_iter)
        val_loss = evaluation_loop.validation_step.step()
        return float(val_loss)

    def _objective(self, trial: optuna.Trial, context: Context, batch_size_finder: BatchSizeFinder) -> float:
        trial_context = self._trial_context(trial, context)
        batch_size, total_descent_steps = batch_size_finder.find_batch_size(trial_context)
        if batch_size <= 0:
            raise TrialPruned("No memory-feasible batch size for this trial.")
        trial_context = trial_context.fork(
            batch_size=batch_size,
            accumulation_steps=max(trial_context.accumulated_batch_size // batch_size, 1),
            descent_steps=total_descent_steps,
        )
        val_loss = self._evaluate_trial(trial_context)
        trial.set_user_attr("batch_size", int(batch_size))
        trial.set_user_attr("descent_steps", int(total_descent_steps))
        trial.report(val_loss, step=1)
        if trial.should_prune():
            raise TrialPruned("Pruned due to underperforming validation loss.")
        return val_loss

    def run(self, context: Context, batch_size_finder: BatchSizeFinder) -> Context:
        sampler = optuna.samplers.TPESampler(seed=context.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.startup_trials, n_warmup_steps=0)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self._objective(trial, context, batch_size_finder), n_trials=self.n_trials)

        best_params = study.best_trial.params
        best_context = context.fork(
            learning_rate=float(best_params["learning_rate"]),
            accumulated_batch_size=int(best_params["accumulated_batch_size"]),
            num_layers=int(best_params["num_layers"]),
        )
        if "vocab_size" in best_params:
            best_context = best_context.fork(vocab_size=int(best_params["vocab_size"]))
        final_batch_size, total_descent_steps = batch_size_finder.find_batch_size(best_context)
        if final_batch_size <= 0:
            raise ValueError("Best Optuna trial is not memory-feasible on final reconstruction.")
        print(f"Optuna best trial value={study.best_value}, params={best_params}")
        return best_context.fork(
            batch_size=final_batch_size,
            accumulation_steps=max(best_context.accumulated_batch_size // final_batch_size, 1),
            descent_steps=total_descent_steps,
        )


class OptunaSearchFactory(Factory[OptunaSearch]):
    type: Literal["optunasearch"] = "optunasearch"

    n_trials: int
    sweep_time: float
    min_lr: float
    max_lr: float
    accumulated_batch_sizes: list[int]
    min_num_layers: int
    max_num_layers: int
    vocab_sizes: list[int] = []
    startup_trials: int = 5

    def build(self, ctx: Context) -> OptunaSearch:
        return OptunaSearch(
            n_trials=self.n_trials,
            sweep_time=self.sweep_time,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            accumulated_batch_sizes=self.accumulated_batch_sizes,
            min_num_layers=self.min_num_layers,
            max_num_layers=self.max_num_layers,
            vocab_sizes=self.vocab_sizes,
            startup_trials=self.startup_trials,
        )

class Pretrainer:
    def __init__(self,
                 tokens_per_param: int|float,
                 training_time: int|float,
                 warmup_steps: int,
                 batch_size_finder: BatchSizeFinder,
                 optuna_search: OptunaSearch,):
        
        self.tokens_per_param = tokens_per_param
        self.training_time = training_time
        self.warmup_steps = warmup_steps
        
        self.batch_size_finder = batch_size_finder
        self.optuna_search = optuna_search
        
    def run(self, context: Context):
        context = context.fork(
            tokens_per_param=self.tokens_per_param,
            training_time=self.training_time,
            warmup_steps=self.warmup_steps,
        )
        print("Running Optuna search.")
        context = self.optuna_search.run(context, self.batch_size_finder)
        print("Optuna search complete.")
        return context
    
class PretrainerFactory(Factory[Pretrainer]):
    type: Literal["pretrainer"] = "pretrainer"
    
    tokens_per_param: int | float
    training_time: int | float
    warmup_steps: int
    batch_size_finder: BatchSizeFinderFactory
    optuna_search: OptunaSearchFactory
    
    def build(self, ctx: Context) -> Pretrainer:
        batch_size_finder = self.batch_size_finder.build(ctx)
        optuna_search = self.optuna_search.build(ctx)
        return Pretrainer(tokens_per_param=self.tokens_per_param,
                          training_time=self.training_time,
                          warmup_steps=self.warmup_steps,
                          batch_size_finder=batch_size_finder,
                          optuna_search=optuna_search)