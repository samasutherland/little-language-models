from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

import torch
from torch import nn
from lib.components.base import BuildContext



# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    ctx: BuildContext = BuildContext()
    vocab_size: int = 4000
    padding_idx: int = 3


    def build(self) -> nn.Module:
        return nn.Embedding(self.vocab_size, self.ctx.embedding_dim, padding_idx=self.padding_idx)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]