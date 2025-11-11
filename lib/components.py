from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter
from torch import nn
from lib.models import positional_encodings, attention_layers, transformer_layers, language_models, activations


class BuildContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    embedding_dim: int = 256
    max_context: int = 512


# ---- Positional Encodings ---- #
class RoPEFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rope"] = "rope"
    base: int = 10000

    def build(self, ctx: BuildContext, qk_dim: int) -> nn.Module:
        return positional_encodings.RoPE(ctx.max_context, qk_dim, base=self.base)


PositionalEncoding = Annotated[Union[RoPEFactory], Field(discriminator="type")]


# ---- Activations ---- #
class IdentityFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["identity"] = "identity"

    def build(self) -> nn.Module:
        return nn.Identity()


class GELUFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["gelu"] = "gelu"

    def build(self) -> nn.Module:
        return nn.GELU()


class RELUFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["relu"] = "relu"

    def build(self) -> nn.Module:
        return nn.ReLU()


class SVDTruncationFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["svdtruncation"] = "svdtruncation"
    eps: float | None = 0.01
    k: int | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.eps is None and self.k is None:
            raise ValueError("SVDTruncation: specify either eps or k")

    def build(self) -> nn.Module:
        return activations.SVDTruncation(eps=self.eps, k=self.k)


class QRTruncationFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["qrtruncation"] = "qrtruncation"
    k: int

    def build(self) -> nn.Module:
        return activations.QRTruncation(k=self.k)


class SVDEntropicReductionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["svdentropicreduction"] = "svdentropicreduction"
    alpha: float

    def build(self) -> nn.Module:
        return activations.SVDEntropicReduction(alpha=self.alpha)


Activation = Annotated[
    Union[
        IdentityFactory, GELUFactory, RELUFactory, QRTruncationFactory, SVDTruncationFactory, SVDEntropicReductionFactory], Field(
        discriminator="type")]


# ---- Attention Layers ---- #
class MultiHeadSelfAttentionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["multiheadselfattention"] = "multiheadselfattention"

    qk_dim: int = 64
    v_dim: int = 64
    causal: bool = True
    n_heads: int = 6
    sdpa: bool = True
    dropout: float = 0.1
    reproject: bool = True

    def build(self, ctx: BuildContext, positional_encoding: PositionalEncoding) -> nn.Module:
        posenc = positional_encoding.build(ctx, self.qk_dim)
        return attention_layers.MultiHeadSelfAttention(posenc,
                                                       embedding_dim=ctx.embedding_dim,
                                                       qk_dim=self.qk_dim,
                                                       v_dim=self.v_dim,
                                                       causal=self.causal,
                                                       max_context=ctx.max_context,
                                                       n_heads=self.n_heads,
                                                       sdpa=self.sdpa,
                                                       dropout=self.dropout,
                                                       reproject=self.reproject)


class LatentMultiHeadSelfAttentionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["multiheadselfattention"] = "multiheadselfattention"

    projection_dim: int
    qk_dim: int = 64
    v_dim: int = 64
    causal: bool = True
    n_heads: int = 6
    sdpa: bool = True
    dropout: float = 0.1
    reproject: bool = True

    def build(self, ctx: BuildContext, positional_encoding: PositionalEncoding) -> nn.Module:
        posenc = positional_encoding.build(ctx, self.qk_dim)
        return attention_layers.LatentMultiHeadSelfAttention(posenc,
                                                             projection_dim=self.projection_dim,
                                                             embedding_dim=ctx.embedding_dim,
                                                             qk_dim=self.qk_dim,
                                                             v_dim=self.v_dim,
                                                             causal=self.causal,
                                                             max_context=ctx.max_context,
                                                             n_heads=self.n_heads,
                                                             sdpa=self.sdpa,
                                                             dropout=self.dropout,
                                                             reproject=self.reproject)


Attention = Annotated[
    Union[MultiHeadSelfAttentionFactory, LatentMultiHeadSelfAttentionFactory], Field(discriminator="type")]


# ---- Transformer Layers ---- #
