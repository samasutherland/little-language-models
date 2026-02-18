from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

from torch import nn

from lib.components.attention_layers import AttentionFactory
from lib.components.activations import ActivationFactory, IdentityFactory
from lib.components.norms import NormFactory
from lib.components.base import BuildContext

# ---------- Layer Definitions ---------- #

class StandardTransformerLayer(nn.Module):
    def __init__(self,
                 activation_factory: ActivationFactory,
                 norm_factory: NormFactory,
                 attention_factory: AttentionFactory,
                 embedding_dim: int,
                 dropout: float,
                 feedforward_dim: int
                 ):
        super().__init__()

        self.norm = norm_factory.build()

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.attention = attention_factory.build()

        if type(activation_factory) == IdentityFactory:
            self.ffn = nn.Identity()  # Identity activation collapses ff layers to no-op.
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dim, feedforward_dim),
                activation_factory.build(),
                self.ffn_dropout,
                nn.Linear(feedforward_dim, embedding_dim)
            )

    def forward(self, x):
        x = x[:, -self.attention.max_context:, :]
        x = x + self.attn_dropout(self.attention(self.norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.norm(x)))
        return x

class StandardTransformerLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardtransformerlayer"] = "standardtransformerlayer"

    activation_factory: ActivationFactory
    norm_factory: NormFactory
    attention_factory: AttentionFactory

    ctx: BuildContext
    dropout: float
    feedforward_dim: int


    def build(self) -> nn.Module:

        return StandardTransformerLayer(self.activation_factory,
                                 self.norm_factory,
                                 self.attention_factory,
                                 embedding_dim = self.ctx.embedding_dim,
                                 dropout = self.dropout,
                                 feedforward_dim = self.feedforward_dim
                                 )

# ---------- Layer Registration ---------- #

TransformerLayerFactory = Annotated[
    Union[StandardTransformerLayerFactory], Field(discriminator="type")]
