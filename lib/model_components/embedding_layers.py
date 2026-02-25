from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field

from torch import nn

from lib.model_components.context import BuildContext


# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    def build(self, ctx: BuildContext) -> nn.Module:
        vocab_size = ctx.require("vocab_size")
        padding_idx = ctx.require("padding_idx")
        embedding_dim = ctx.require("embedding_dim")

        return nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]