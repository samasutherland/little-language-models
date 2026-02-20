from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field

from torch import nn

from lib.components.base import BuildContext


# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    vocab_size: int
    padding_idx: int

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.Embedding(self.vocab_size, ctx.embedding_dim, padding_idx=self.padding_idx)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]