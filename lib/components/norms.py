from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, Field

from torch import nn

from lib.components.base import BuildContext


# ---------- Layer Definitions ---------- #

class IdentityFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["identity"] = "identity"

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.Identity()

class RMSNormFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rmsnorm"] = "rmsnorm"

    dim: Optional[int] = None

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.RMSNorm(self.dim if self.dim is not None else ctx.embedding_dim)


class LayerNormFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["layernorm"] = "layernorm"

    dim: Optional[int] = None

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.LayerNorm(self.dim if self.dim is not None else ctx.embedding_dim)

# ---------- Layer Registration ---------- #

NormFactory = Annotated[
    Union[
        IdentityFactory, RMSNormFactory, LayerNormFactory], Field(
        discriminator="type")]