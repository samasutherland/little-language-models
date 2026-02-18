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

    embedding_dim: int

    transformer_layer_factory: TransformerLayerFactory
    final_norm_factory: NormFactory
    embedding_layer_factory: EmbeddingLayerFactory
    num_layers: int

    @model_validator(mode="after")
    def _apply_embedding_dim(self):
        # Ensure sub-factories that depend on embedding_dim receive it explicitly.
        if hasattr(self.embedding_layer_factory, "embedding_dim") and self.embedding_layer_factory.embedding_dim is None:
            self.embedding_layer_factory.embedding_dim = self.embedding_dim

        if hasattr(self.transformer_layer_factory, "embedding_dim") and self.transformer_layer_factory.embedding_dim is None:
            self.transformer_layer_factory.embedding_dim = self.embedding_dim

        # Norm factories may optionally use embedding_dim as their default dimension.
        if isinstance(self.final_norm_factory, RMSNormFactory) and getattr(self.final_norm_factory, "embedding_dim", None) is None:
            self.final_norm_factory.embedding_dim = self.embedding_dim

        return self

    def build(self) -> nn.Module:
        return Transformer(
            self.transformer_layer_factory,
            self.final_norm_factory,
            self.embedding_layer_factory,
            self.num_layers,
        )

# ---------- Layer Registration ---------- #

LanguageModelFactory = Annotated[Union[TransformerFactory], Field(discriminator="type")]