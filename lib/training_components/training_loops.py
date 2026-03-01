from typing import Literal, Annotated, Union, Any
from pydantic import Field

from lib import Context, Factory


import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lib.training_components import OptimizerFactory



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

                 experiment_name: str,
                 ):

        self.loss_buffer = Buffer(100)

    def step(self,):
