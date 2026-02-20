from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field

from torch import nn

from lib.components.base import BuildContext
from lib.components.attention_layers import AttentionFactory
from lib.components.activations import ActivationFactory, IdentityFactory
from lib.components.norms import NormFactory


# ---------- Layer Definitions ---------- #

class StandardTransformerLayer(nn.Module):
    def __init__(self,
                 activation: nn.Module,
                 attention_norm: nn.Module,
                 feedforward_norm: nn.Module,
                 attention: nn.Module,
                 embedding_dim: int,
                 dropout: float,
                 feedforward_dim: int,
                 ):
        super().__init__()

        self.attention_norm = attention_norm
        self.feedforward_norm = feedforward_norm

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.attention = attention

        if isinstance(activation, nn.Identity):
            self.ffn = nn.Identity()  # Identity activation collapses ff layers to no-op.
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dim, feedforward_dim),
                activation,
                self.ffn_dropout,
                nn.Linear(feedforward_dim, embedding_dim)
            )

    def forward(self, x):
        x = x[:, -self.attention.max_context:, :]
        x = x + self.attn_dropout(self.attention(self.attention_norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.feedforward_norm(x)))
        return x

class StandardTransformerLayerFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["standardtransformerlayer"] = "standardtransformerlayer"

    activation_factory: ActivationFactory
    attention_norm_factory: NormFactory
    feedforward_norm_factory: NormFactory
    attention_factory: AttentionFactory

    dropout: float
    feedforward_dim: int

    def build(self, ctx: BuildContext) -> nn.Module:
        activation = self.activation_factory.build(ctx)
        attention_norm = self.attention_norm_factory.build(ctx)
        feedforward_norm = self.feedforward_norm_factory.build(ctx)
        attention = self.attention_factory.build(ctx)
        
        return StandardTransformerLayer(
            activation,
            attention_norm,
            feedforward_norm,
            attention,
            embedding_dim=ctx.embedding_dim,
            dropout=self.dropout,
            feedforward_dim=self.feedforward_dim,
        )

# ---------- Layer Registration ---------- #

TransformerLayerFactory = Annotated[
    Union[StandardTransformerLayerFactory], Field(discriminator="type")]
