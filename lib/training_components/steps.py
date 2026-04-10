from typing import Literal, Annotated, Union, Any
from pydantic import Field
from torch.utils.data import DataLoader

from lib import Context, Factory


import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lib.training_components import OptimizerFactory, CriterionFactory, SchedulerFactory
from lib.data_components import DataLoaderFactory

class EvaluationStep:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 autocast_ctx: torch.autocast,
                 device: torch.device):
                 # logger:):
        self.model = model
        self.criterion = criterion
        self.autocast_ctx = autocast_ctx
        self.device = device

    def step(self, x: torch.Tensor) -> torch.Tensor:
        self.model.train()
        x = x.to(self.device, non_blocking=True)
        with self.autocast_ctx:
            logits = self.model(x[:, :-1])
            targets = x[:, 1:][:, -logits.shape[1]:]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss

class EvaluationStepFactory(Factory[EvaluationStep]):
    type: Literal["evaluationstep"] = "evaluationstep"
    
    criterion_factory: CriterionFactory

    def build(self, ctx: Context) -> EvaluationStep:
        criterion = self.criterion_factory.build(ctx)

        model = ctx.require("model")
        autocast_ctx = ctx.require("autocast_ctx")
        device = ctx.require("device")

        return EvaluationStep(model,
                              criterion,
                              autocast_ctx,
                              device)


class GradientStep:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 grad_clip_norm: float
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.step_scheduler = True

    def step(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            if self.step_scheduler:
                try:
                    self.scheduler.step()
                except ValueError:
                    print("Scheduler stopped stepping.")
                    self.step_scheduler = False
            self.optimizer.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_last_lr()[0]

class GradientStepFactory(Factory[GradientStep]):
    type: Literal["gradientstep"] = "gradientstep"

    scheduler_factory: SchedulerFactory

    grad_clip_norm: float

    def build(self, ctx: Context) -> GradientStep:
        model = ctx.require("model")

        scheduler = self.scheduler_factory.build(ctx)
        optimizer = scheduler.optimizer

        return GradientStep(model,
                            optimizer,
                            scheduler,
                            grad_clip_norm=self.grad_clip_norm)


class ValidationStep:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 device: torch.device,
                 data_loader: DataLoader,
                 num_batches: int):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.data_loader = data_loader
        self.num_batches = num_batches

    def step(self) -> float:
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            data_iter = iter(self.data_loader)
            for i in range(self.num_batches):
                val_batch = next(data_iter)
                x = val_batch.to(self.device, non_blocking=True)
                logits = self.model(x[:, :-1])
                targets = x[:, 1:][:, -logits.shape[1]:]
                val_loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_losses.append(val_loss.item())

            mean_val_loss = torch.tensor(val_losses).mean().item()

        return mean_val_loss

class ValidationStepFactory(Factory[ValidationStep]):
    type: Literal["validationstep"] = "validationstep"

    criterion_factory: CriterionFactory

    validation_batches: int

    def build(self, ctx: Context) -> ValidationStep:
        model = ctx.require("model")
        device = ctx.require("device")
        data_loader = ctx.require("val_dataloader")

        criterion = self.criterion_factory.build(ctx)

        return ValidationStep(model,
                              criterion,
                              device,
                              data_loader=data_loader,
                              num_batches=self.validation_batches,
                              )


StepFactory = Annotated[
    Union[EvaluationStepFactory, GradientStepFactory, ValidationStepFactory],
    Field(discriminator="type"),
]
