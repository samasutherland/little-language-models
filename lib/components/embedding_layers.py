from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

import torch
from torch import nn

# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    embedding_dim: int

    vocab_size: int
    padding_idx: int


    def build(self) -> nn.Module:
        return nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]