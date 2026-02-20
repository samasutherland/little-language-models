from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field

import torch
from torch import nn

from lib.components.base import BuildContext
from lib.components.transformer_layers import TransformerLayerFactory
from lib.components.norms import NormFactory
from lib.components.embedding_layers import EmbeddingLayerFactory


# ---------- Layer Definitions ---------- #

class Transformer(nn.Module):
    def __init__(self,
                 embedding: nn.Module,
                 transformer_stack: nn.Sequential,
                 final_norm: nn.Module,
                 ):
        super().__init__()

        self.embedding = embedding
        self.transformer_stack = transformer_stack
        self.final_norm = final_norm

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.t())
        return logits

class TransformerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["transformer"] = "transformer"

    transformer_layer_factory: TransformerLayerFactory
    final_norm_factory: NormFactory
    embedding_layer_factory: EmbeddingLayerFactory
    num_layers: int

    def build(self, ctx: BuildContext) -> nn.Module:
        embedding = self.embedding_layer_factory.build(ctx)
        transformer_stack = nn.Sequential(
            *[self.transformer_layer_factory.build(ctx) for _ in range(self.num_layers)]
        )
        final_norm = self.final_norm_factory.build(ctx)
        
        return Transformer(
            embedding,
            transformer_stack,
            final_norm,
        )

# ---------- Layer Registration ---------- #

LanguageModelFactory = Annotated[Union[TransformerFactory], Field(discriminator="type")]
