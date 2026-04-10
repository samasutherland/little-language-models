from typing import Literal, Annotated, Union
from pydantic import ConfigDict, Field

import torch
from torch import nn

from lib import Context, Factory
from lib.model_components.transformer_layers import TransformerLayerFactory
from lib.model_components.norms import NormFactory
from lib.model_components.embedding_layers import EmbeddingLayerFactory


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

class TransformerFactory(Factory[nn.Module]):
    
    type: Literal["transformer"] = "transformer"

    transformer_layer_factory: TransformerLayerFactory
    final_norm_factory: NormFactory
    embedding_layer_factory: EmbeddingLayerFactory

    embedding_dim: int

    def build(self, ctx: Context) -> nn.Module:
        num_layers = ctx.require("num_layers")
        
        ctx_fork = ctx.fork(embedding_dim=self.embedding_dim)
        embedding = self.embedding_layer_factory.build(ctx_fork)
        transformer_stack = nn.Sequential(
            *[self.transformer_layer_factory.build(ctx_fork) for _ in range(num_layers)]
        )
        final_norm = self.final_norm_factory.build(ctx_fork)
        
        return Transformer(
            embedding,
            transformer_stack,
            final_norm,
        )

# ---------- Layer Registration ---------- #

LanguageModelFactory = Annotated[Union[TransformerFactory], Field(discriminator="type")]
