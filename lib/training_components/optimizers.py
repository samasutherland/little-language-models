from typing import Literal, Annotated, Union, Any
from pydantic import Field

from lib import Context, Factory

import torch
from torch import nn
from torch.optim import Optimizer

class OptimizerFactoryBase(Factory[Optimizer]):
    @staticmethod
    def collect_parameters(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        parameters = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return parameters


class AdamWFactory(OptimizerFactoryBase):
    type: Literal["adamw"] = "adamw"
    weight_decay: float
    betas: tuple[float, float]

    def build(self, ctx: Context) -> Optimizer:
        model = ctx.require("model")
        lr = ctx.require("lr")

        parameters = self.collect_parameters(model, self.weight_decay)

        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=self.betas,
        )

class AdamFactory(OptimizerFactoryBase):
    type: Literal["adam"] = "adam"
    weight_decay: float
    betas: tuple[float, float]

    def build(self, ctx: Context) -> Optimizer:
        model = ctx.require("model")
        lr = ctx.require("lr")

        parameters = self.collect_parameters(model, self.weight_decay)

        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=self.betas,
        )

class SGDFactory(OptimizerFactoryBase):
    type: Literal["sgd"] = "sgd"
    weight_decay: float
    momentum: float

    def build(self, ctx: Context) -> Optimizer:
        model = ctx.require("model")
        lr = ctx.require("lr")

        parameters = self.collect_parameters(model, self.weight_decay)

        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=self.momentum,
        )


OptimizerFactory = Annotated[
    Union[AdamWFactory, AdamFactory, SGDFactory],
    Field(discriminator="type"),
]
