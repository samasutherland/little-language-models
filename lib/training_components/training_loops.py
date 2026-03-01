import warnings
from typing import Literal, Annotated, Union, Any
from pydantic import Field
from torch.utils.data import DataLoader

from lib import Context, Factory


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pathlib import Path
import os

from aim import Run

from lib.training_components import OptimizerFactory
from lib.training_components.steps import EvaluationStep, GradientStep, ValidationStep


class Buffer: # TODO: subclass abstract base class, or use existing first in first out structure
    def __init__(self, length):
        self.length = length
        self.buffer = torch.zeros(length)

    def push(self, x):
        torch.cat((self.buffer[1:], torch.tensor([x])))

    def reset(self):
        self.buffer = torch.zeros(self.length)






class TrainingLoop:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 criterion: nn.Module,
                 device: torch.device,

                 dataloader: DataLoader,

                 autocast_ctx: torch.autocast,

                 logger: Run,
                 descent_steps: int,
                 accumulation_steps: int,

                 evaluation_step: EvaluationStep,
                 gradient_step: GradientStep,
                 validation_step: ValidationStep,

                 save_dir: str|Path

                 ):
        self.model = model
        self.loss_buffer = Buffer(100)
        self.device = device
        self.autocast_ctx = autocast_ctx
        self.dataloader = iter(dataloader)

        self.descent_steps = descent_steps
        self.accumulation_steps = accumulation_steps

        self.logger = logger

        self.evaluation_step = evaluation_step
        self.gradient_step = gradient_step
        self.validation_step = validation_step

        self.token_count = 0
        self.best_loss = float('inf')

        self.save_dir = Path(save_dir)



    def run(self,):
        for i in range(self.descent_steps):
            batch_loss = 0.0
            for j in range(self.accumulation_steps):
                batch = next(self.dataloader)
                x = batch["input_ids"]
                self.token_count += x[:, 1:].numel()
                loss = self.evaluation_step.step(x)

                if loss.item() > 10 * torch.median(self.loss_buffer.buffer):
                    warnings.warn("loss huge. skipping...")
                    continue

                loss_buffer = loss_buffer.push(loss.item())
                loss = loss / self.accumulation_steps

                batch_loss += loss.item()

                loss.backward()

            self.logger.track(batch_loss, name="loss", step=i, context={"subset": "train"})
            if not torch.isfinite(loss):
                pass
            else:
                self.logger.track(torch.exp(torch.tensor(batch_loss).clamp(max=88.72)).item(), name="perplexity", step=i,
                          context={"subset": "train"})
            self.logger.track(self.gradient_step.scheduler.get_last_lr()[0], name="lr", step=i, context={"subset": "train"})

            if batch_loss < best_loss:
                best_loss = batch_loss
                torch.save({"model": self.model.state_dict(), "optimizer": self.gradient_step.optimizer.state_dict(),
                            "step": i}, os.path.join(self.save_dir, "ckpt_best.pt"))
                with open(os.path.join(self.save_dir, "best_loss_step.txt"), "w") as f:
                    f.write(f"loss of {best_loss} achieved on step {i}")
            self.logger.track(best_loss, name="best_loss", step=i, context={"subset": "train"})


            self.gradient_step.step()