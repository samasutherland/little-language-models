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
