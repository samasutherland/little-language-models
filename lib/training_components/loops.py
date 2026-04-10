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

from aim import Run
import yaml

from lib.training_components import OptimizerFactory
from lib.training_components.steps import EvaluationStep, GradientStep, ValidationStep, StepFactory
from lib.training_components.loggers import LoggerFactory, Checkpointer, AimLogger, NullLoggerFactory, \
    NullCheckpointerFactory


class Buffer: # TODO: subclass abstract base class, or use existing first in first out structure
    def __init__(self, length):
        self.length = length
        self.buffer = torch.full((length,), float('inf'))

    def push(self, x):
        self.buffer = torch.cat((self.buffer[1:], torch.tensor([x])))

    def reset(self):
        self.buffer = torch.zeros(self.length)


class TrainingLoop:
    def __init__(self,
                 dataloader: DataLoader,

                 descent_steps: int,
                 accumulation_steps: int,
                 val_frequency: int,

                 evaluation_step: EvaluationStep,
                 gradient_step: GradientStep,
                 validation_step: ValidationStep,


                 aim_logger: AimLogger,
                 train_checkpointer: Checkpointer,
                 val_checkpointer: Checkpointer,

                 ):
        self.loss_buffer = Buffer(100)
        self.dataloader = iter(dataloader)

        self.descent_steps = descent_steps
        self.accumulation_steps = accumulation_steps
        assert self.accumulation_steps > 0

        self.val_frequency = val_frequency

        self.evaluation_step = evaluation_step
        self.gradient_step = gradient_step
        self.validation_step = validation_step

        self.token_count = 0
        self.aim_logger = aim_logger
        self.train_checkpointer = train_checkpointer
        self.val_checkpointer = val_checkpointer



    def run(self,):
        for i in tqdm(range(self.descent_steps)):
            batch_loss = 0.0
            for j in range(self.accumulation_steps):
                x = next(self.dataloader)
                self.token_count += x[:, 1:].numel()
                loss = self.evaluation_step.step(x)

                if loss.item() > 10 * torch.median(self.loss_buffer.buffer):
                    warnings.warn("loss huge. skipping...")
                    continue

                self.loss_buffer.push(loss.item())
                loss = loss / self.accumulation_steps

                batch_loss += loss.item()

                loss.backward()

            train_metrics = {"loss": batch_loss}
            if torch.isfinite(torch.tensor(batch_loss)):
                train_metrics["perplexity"] = torch.exp(torch.tensor(batch_loss).clamp(max=88.72)).item()
            train_metrics["lr"] = self.gradient_step.lr

            self.aim_logger.track_train_metrics(train_metrics, i)
            self.train_checkpointer.compare_loss_and_checkpoint(i, batch_loss)

            self.gradient_step.step()

            if i % self.val_frequency == 0:
                val_loss = self.validation_step.step()
                self.aim_logger.track_val_metrics({"loss": val_loss}, i)
                self.val_checkpointer.compare_loss_and_checkpoint(i, val_loss)
                
        total_descent_steps = i + 1

        return self.token_count, loss, val_loss, self.train_checkpointer.best_loss, self.val_checkpointer.best_loss, total_descent_steps

class TrainingLoopFactory(Factory[TrainingLoop]):
    type: Literal["trainingloop"] = "trainingloop"

    evaluation_step_factory: StepFactory
    gradient_step_factory: StepFactory
    validation_step_factory: StepFactory

    aim_logger_factory: LoggerFactory
    train_checkpointer_factory: LoggerFactory
    val_checkpointer_factory: LoggerFactory

    def build(self, ctx: Context) -> TrainingLoop:
        
        dataloader = ctx.require("train_dataloader")

        ctx = ctx.fork(pad_id=dataloader.dataset.pad_id)
        
        descent_steps = ctx.require("descent_steps")
        accumulation_steps = ctx.require("accumulation_steps")
        val_frequency = ctx.require("val_frequency")

        evaluation_step = self.evaluation_step_factory.build(ctx)
        gradient_step = self.gradient_step_factory.build(ctx)
        validation_step = self.validation_step_factory.build(ctx)

        aim_logger = self.aim_logger_factory.build(ctx)
        
        ctx_fork = ctx.fork(optimizer=gradient_step.optimizer)
        train_checkpointer = self.train_checkpointer_factory.build(ctx_fork)
        val_checkpointer = self.val_checkpointer_factory.build(ctx_fork)

        return TrainingLoop(dataloader,
                            descent_steps=descent_steps,
                            accumulation_steps=accumulation_steps,
                            val_frequency=val_frequency,
                            evaluation_step=evaluation_step,
                            gradient_step=gradient_step,
                            validation_step=validation_step,
                            aim_logger=aim_logger,
                            train_checkpointer=train_checkpointer,
                            val_checkpointer=val_checkpointer,)


class BenchmarkingLoopFactory(Factory[TrainingLoop]):
    model_config = ConfigDict(extra="ignore")
    type: Literal["benchmarkingloop"] = "benchmarkingloop"

    evaluation_step_factory: StepFactory
    gradient_step_factory: StepFactory
    validation_step_factory: StepFactory

    # No checkpointers or loggers for benchmarking
    # aim_logger_factory: LoggerFactory
    # train_checkpointer_factory: LoggerFactory
    # val_checkpointer_factory: LoggerFactory

    # context_path: str | Path

    def build(self, ctx: Context) -> TrainingLoop:
        dataloader = ctx.require("train_dataloader")
        ctx = ctx.fork(pad_id=dataloader.dataset.pad_id)

        descent_steps = ctx.require("descent_steps")
        accumulation_steps = ctx.require("accumulation_steps")
        val_frequency = ctx.require("val_frequency")

        evaluation_step = self.evaluation_step_factory.build(ctx)
        gradient_step = self.gradient_step_factory.build(ctx)
        validation_step = self.validation_step_factory.build(ctx)

        aim_logger = NullLoggerFactory().build(ctx)

        ctx_fork = ctx.fork(optimizer=gradient_step.optimizer)
        train_checkpointer = NullCheckpointerFactory().build(ctx_fork)
        val_checkpointer = NullCheckpointerFactory().build(ctx_fork)

        return TrainingLoop(dataloader,
                            descent_steps=descent_steps,
                            accumulation_steps=accumulation_steps,
                            val_frequency=val_frequency,
                            evaluation_step=evaluation_step,
                            gradient_step=gradient_step,
                            validation_step=validation_step,
                            aim_logger=aim_logger,
                            train_checkpointer=train_checkpointer,
                            val_checkpointer=val_checkpointer, )

LoopFactory = Annotated[
    Union[TrainingLoopFactory, BenchmarkingLoopFactory],
    Field(discriminator="type"),
]