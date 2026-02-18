from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

import torch
from torch import nn
from lib.components.transformer_layers import TransformerLayerFactory, StandardTransformerLayerFactory
from lib.components.norms import NormFactory, RMSNormFactory
from lib.components.embedding_layers import EmbeddingLayerFactory, StandardEmbeddingLayerFactory


# ---------- Layer Definitions ---------- #

class Transformer(nn.Module):
    def __init__(self,
                 transformer_layer_factory: TransformerLayerFactory,
                 final_norm_factory: NormFactory,
                 embedding_layer_factory: EmbeddingLayerFactory,
                 num_layers: int,
                 ):
        super().__init__()

        self.embedding = embedding_layer_factory.build()
        self.transformer_stack = nn.Sequential(*[transformer_layer_factory.build() for _ in range(num_layers)])

        self.final_norm = final_norm_factory.build()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.t())
        return logits

class TransformerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["transformer"] = "transformer"

    transformer_layer_factory: TransformerLayerFactory = StandardTransformerLayerFactory()
    final_norm_factory: NormFactory = RMSNormFactory()
    embedding_layer_factory: EmbeddingLayerFactory = StandardEmbeddingLayerFactory()
    num_layers: int = 4

    def build(self) -> nn.Module:
        return Transformer(self.transformer_layer_factory,
                           self.final_norm_factory,
                           self.embedding_layer_factory,
                           self.num_layers,
                           )

# ---------- Layer Registration ---------- #

LanguageModelFactory = Annotated[Union[TransformerFactory], Field(discriminator="type")]