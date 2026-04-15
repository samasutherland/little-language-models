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
import os
from pathlib import Path


class AimLogger(Run):
    def __init__(self, experiment_name: str, configs: dict):
        super().__init__(experiment=experiment_name)
        for key, value in configs.items():
            self[key] = value

    def track_metrics(self, metrics, step, context):
        for key, value in metrics.items():
            self.track(value, name=key, step=step, context=context)

    def track_train_metrics(self, metrics, step):
        self.track_metrics(metrics, step, {"subset": "train"})

    def track_val_metrics(self, metrics, step):
        self.track_metrics(metrics, step, {"subset": "val"})


class AimLoggerFactory(Factory[Run]):
    type: Literal["aimloggerfactory"] = "aimloggerfactory"

    def build(self, ctx: Context) -> AimLogger:
        configs = ctx.require("config_dicts")
        experiment_name = ctx.require("experiment_name")
        
        return AimLogger(experiment_name=experiment_name,
                         configs=configs)

class Checkpointer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_dir: str | Path,
        checkpoint_filename: str,
    ):
        self.save_dir = str(save_dir)
        self.checkpoint_filename = checkpoint_filename
        self.best_loss = float("inf")
        self.model = model
        self.optimizer = optimizer
        
        os.makedirs(self.save_dir, exist_ok=True)
       

    def compare_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def save_checkpoint(self, step, loss):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "step": step, "loss": loss}, os.path.join(self.save_dir, self.checkpoint_filename))

    def compare_loss_and_checkpoint(self, step, loss):
        if self.compare_loss(loss):
            self.save_checkpoint(step, loss)

class CheckpointerFactory(Factory[Checkpointer]):
    type: Literal["checkpointerfactory"] = "checkpointerfactory"

    folder_name: str | Path
    checkpoint_root: str | Path = "data/"
    checkpoint_filename: str

    def build(self, ctx: Context) -> Checkpointer:
        model = ctx.require("model")
        optimizer = ctx.require("optimizer")
        save_dir = os.path.join(self.checkpoint_root, str(self.folder_name))
        return Checkpointer(model,
                            optimizer,
                            save_dir=save_dir,
                            checkpoint_filename=self.checkpoint_filename,
                            )
    
class NullCheckpointer(Checkpointer):
    def __init__(self):
        self.best_loss = float("inf")
        return

    def compare_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def save_checkpoint(self, step, loss):
        return

    def compare_loss_and_checkpoint(self, step, loss):
        self.compare_loss(loss)
    
class NullCheckpointerFactory(Factory[Checkpointer]):
    type: Literal["nullcheckpointerfactory"] = "nullcheckpointerfactory"
    def build(self, ctx: Context) -> NullCheckpointer:
        return NullCheckpointer()

class NullLogger(AimLogger):
    def __init__(self):
        return
    
    def track_metrics(self, metrics, step, context):
        return

    def track_train_metrics(self, metrics, step):
        return

    def track_val_metrics(self, metrics, step):
        return


class NullLoggerFactory(Factory[NullLogger]):
    type: Literal["nullloggerfactory"] = "nullloggerfactory"

    def build(self, ctx: Context) -> NullLogger:
        return NullLogger()


LoggerFactory = Annotated[
    Union[AimLoggerFactory, CheckpointerFactory, NullLoggerFactory, NullCheckpointerFactory],
    Field(discriminator="type"),
]