from typing import Literal, Annotated, Union, Any
from pydantic import Field
import functools

from aim import Run, Text
from lib import Context, Factory

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lib.training_components import OptimizerFactory
from pathlib import Path
import os

#
class AimLogger(Run):
    def __init__(self, experiment_name: str, configs: dict):
        super().__init__(experiment=experiment_name)
        for key, value in configs.items():
            self[key] = value

    def track_metrics(self, metrics, step, context):
        for key, value in metrics.items():
            self.track(value, name=key, step=step, context=context)

    track_train_metrics = functools.partial(track_metrics, context={"subset": "train"})
    track_val_metrics = functools.partial(track_metrics, context={"subset": "val"})

class AimLoggerFactory(Factory[Run]):
    type: Literal["aimloggerfactory"] = "aimloggerfactory"

    experiment_name: str

    def build(self, ctx: Context) -> AimLogger:
        configs = ctx.require("config_dicts")
        return AimLogger(experiment_name=self.experiment_name,
                         configs=configs)

class Checkpointer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, save_dir: str):
        self.save_dir = save_dir
        self.best_loss = float("inf")
        self.model = model
        self.optimizer = optimizer

    def compare_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def save_checkpoint(self, step, loss):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "step": step, "loss": loss}, os.path.join(Path(self.save_dir), "ckpt_best_val.pt"))

    def compare_loss_and_checkpoint(self, step, loss):
        if self.compare_loss(loss):
            self.save_checkpoint(step, loss)

class CheckpointerFactory(Factory[Checkpointer]):
    type: Literal["checkpointerfactory"] = "checkpointerfactory"

    save_dir: str|Path

    def build(self, ctx: Context) -> Checkpointer:
        model = ctx.require("model")
        optimizer = ctx.require("optimizer")
        return Checkpointer(model,
                            optimizer,
                            save_dir=self.save_dir
                            )

LoggerFactory = Annotated[
    Union[AimLoggerFactory, CheckpointerFactory],
    Field(discriminator="type"),
]