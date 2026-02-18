from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

from torch.nn import Module
from torch import nn
from torch import Tensor
import torch
from functools import cache
from torch.cuda.amp import autocast

from torch.nn import *

# ---------- Layer Definitions ---------- #

class IdentityFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["identity"] = "identity"

    def build(self) -> nn.Module:
        return nn.Identity()

class RMSNormFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rmsnorm"] = "rmsnorm"

    dim: int = 256

    def build(self) -> nn.Module:
        return nn.RMSNorm(self.dim)


class LayerNormFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["layernorm"] = "layernorm"

    dim: int = 256

    def build(self) -> nn.Module:
        return nn.LayerNorm(self.dim)

# ---------- Layer Registration ---------- #

NormFactory = Annotated[
    Union[
        IdentityFactory, RMSNormFactory, LayerNormFactory], Field(
        discriminator="type")]