from typing import Literal, Annotated, Union, Any
from pydantic import Field

from lib import Context, Factory

import torch
from torch.optim.lr_scheduler import LRScheduler

from lib.training_components import OptimizerFactory


class OneCycleLRFactory(Factory[LRScheduler]):
    type: Literal["onecyclelr"] = "onecyclelr"

    optimizer_factory: OptimizerFactory

    initial_div_factor: float # factor to divide lr by to get initial lr
    final_div_factor: float # factor to divide lr by to get final lr
    peak_frac: float # What fraction of the way through the training run the peak learning rate occurs

    def build(self, ctx: Context) -> LRScheduler:
        total_steps = ctx.require("descent_steps")
        lr = ctx.require("learning_rate")

        initial_lr = lr / self.initial_div_factor
        ctx_fork = ctx.fork(lr=initial_lr)
        optimizer = self.optimizer_factory.build(ctx_fork)

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            pct_start=self.peak_frac,
            div_factor=self.initial_div_factor,
            final_div_factor=self.final_div_factor,
            total_steps=total_steps
        )

class CosineAnnealingLRFactory(Factory[LRScheduler]):
    type: Literal["cosineannealing"] = "cosineannealing"

    final_lr: float

    def build(self, ctx: Context) -> LRScheduler:
        total_steps = ctx.require("descent_steps")
        lr = ctx.require("learning_rate")

        ctx_fork = ctx.fork(lr=lr)
        optimizer = self.optimizer_factory.build(ctx_fork)

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.final_lr,
        )

SchedulerFactory = Annotated[
    Union[OneCycleLRFactory, CosineAnnealingLRFactory],
    Field(discriminator="type"),
]