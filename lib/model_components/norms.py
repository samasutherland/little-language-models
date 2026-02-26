from typing import Literal, Annotated, Union, Optional
from pydantic import ConfigDict, Field

from torch import nn

from lib import Factory, Context


# ---------- Layer Definitions ---------- #

class IdentityFactory(Factory[nn.Module]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["identity"] = "identity"

    def build(self, ctx: Context) -> nn.Module:
        return nn.Identity()

class RMSNormFactory(Factory[nn.Module]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rmsnorm"] = "rmsnorm"

    dim: Optional[int] = None

    def build(self, ctx: Context) -> nn.Module:
        embedding_dim = ctx.require("embedding_dim")

        return nn.RMSNorm(self.dim if self.dim is not None else embedding_dim)


class LayerNormFactory(Factory[nn.Module]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["layernorm"] = "layernorm"

    dim: Optional[int] = None

    def build(self, ctx: Context) -> nn.Module:
        embedding_dim = ctx.require("embedding_dim")
        return nn.LayerNorm(self.dim if self.dim is not None else embedding_dim)

# ---------- Layer Registration ---------- #

NormFactory = Annotated[
    Union[
        IdentityFactory, RMSNormFactory, LayerNormFactory], Field(
        discriminator="type")]