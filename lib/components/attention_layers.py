import torch
from torch import nn
from lib.components import positional_encodings
from torch.nn import functional as F
from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

from lib.components.positional_encodings import PositionalEncodingFactory, RoPEFactory
from lib.components.base import BuildContext

# ---------- Layer Definitions ---------- #

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 positional_encoding_factory: PositionalEncodingFactory,
                 embedding_dim: int=384,
                 qk_dim: int=64,
                 v_dim: int=64,
                 causal: bool=True,
                 max_context: int=100,
                 n_heads: int=6,
                 sdpa: bool=False,
                 dropout: float=0.1,
                 reproject=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_context = max_context
        self.v_dim = v_dim
        self.causal = causal
        self.n_heads = n_heads
        self.sdpa = sdpa
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.qk_positional_encoding = positional_encoding_factory.build()
        self.qk_dim = self.qk_positional_encoding.dim

        self.inv_sqrt_qk_dim = 1 / self.qk_dim ** 0.5

        self.q = nn.Linear(embedding_dim, self.qk_dim * n_heads)  # Grouped matrix for the heads
        self.kv = nn.Linear(embedding_dim, (self.qk_dim + v_dim) * n_heads)

        if causal:
            mask = torch.zeros(self.qk_positional_encoding.max_context, self.qk_positional_encoding.max_context)
            self.register_buffer("mask_array",
                                 mask.masked_fill(~torch.tril(torch.ones(self.qk_positional_encoding.max_context, self.qk_positional_encoding.max_context, dtype=torch.bool)),
                                                  float("-inf")))

        if n_heads * v_dim != embedding_dim or reproject:
            self.reproject = nn.Linear(n_heads * v_dim, embedding_dim)
        else:
            self.reproject = nn.Identity()

    def forward(self, x):

        x = x[:, -self.qk_positional_encoding.max_context:, :]
        batch_dim, seq_len, embed_dim = x.shape
        q = self.q(x)  # batch, seq, n_heads * qk_dim

        kv = self.kv(x)  # batch, seq, n_heads * (qk_dim + v_dim)
        k, v = torch.split(kv, [self.qk_dim * self.n_heads, self.v_dim * self.n_heads],
                           dim=-1)  # batch, seq, n_heads * qk_dim; batch, seq, n_heads * v_dim

        # Use view to prevent copying
        q = self.qk_positional_encoding(
            q.view(batch_dim, seq_len, self.n_heads, self.qk_dim).transpose(1, 2))  # batch, n_heads, seq, qk_dim
        k = self.qk_positional_encoding(
            k.view(batch_dim, seq_len, self.n_heads, self.qk_dim).transpose(1, 2))  # batch, n_heads, seq, qk_dim
        v = v.view(batch_dim, seq_len, self.n_heads, self.v_dim).transpose(1,
                                                                           2)  # self.v_rope(v.view(batch_dim, seq_len, self.n_heads, self.v_dim).transpose(1,2)) # batch, n_heads, seq, v_dim

        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.dropout)
        else:

            attn_scores = q @ k.transpose(2, 3) * self.inv_sqrt_qk_dim  # batch, n_heads, seq, seq

            if self.causal:
                attn_scores += self.mask_array[-seq_len:, -seq_len:]

            attn_probs = torch.softmax(attn_scores, dim=-1)  # batch, n_heads, seq, seq
            if self.dropout > 0 and self.training:
                attn_probs = self.dropout_layer(attn_probs)
            attn_output = attn_probs @ v  # batch, n_heads, seq, v_dim

        # TODO: this reshape copies data. view would be better if I can get the shapes to work.

        return self.reproject(attn_output.transpose(1, 2).reshape(batch_dim, seq_len,
                                                                  self.v_dim * self.n_heads))  # batch, seq, embedding_dim

class MultiHeadSelfAttentionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["multiheadselfattention"] = "multiheadselfattention"

    positional_encoding_factory: PositionalEncodingFactory
    ctx: BuildContext

    v_dim: int
    causal: bool
    n_heads: int
    sdpa: bool
    dropout: float
    reproject: bool


    def build(self) -> nn.Module:
        return MultiHeadSelfAttention(self.positional_encoding_factory,
                                                       embedding_dim=self.ctx.embedding_dim,
                                                       v_dim=self.v_dim,
                                                       causal=self.causal,
                                                       n_heads=self.n_heads,
                                                       sdpa=self.sdpa,
                                                       dropout=self.dropout,
                                                       reproject=self.reproject)


class LatentMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self,
                 positional_encoding_factory: PositionalEncodingFactory,
                 embedding_dim: int=384,
                 projection_dim: int=128,
                 v_dim: int=64,
                 causal: bool=True,
                 n_heads: int=6,
                 sdpa: bool=False,
                 dropout: float=0.1,
                 reproject=True):
        super().__init__(positional_encoding_factory, embedding_dim=embedding_dim, v_dim=v_dim, causal=causal,
                        n_heads=n_heads, sdpa=sdpa, dropout=dropout,reproject=reproject)

        # Overwrite kv transform with latent space version
        self.kv = nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.Linear(projection_dim, (
                    self.qk_dim + v_dim) * n_heads))  # Grouped matrix for the heads and k and v

class LatentMultiHeadSelfAttentionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["latentmultiheadselfattention"] = "latentmultiheadselfattention"

    positional_encoding_factory: PositionalEncodingFactory
    ctx: BuildContext

    projection_dim: int
    v_dim: int
    causal: bool
    n_heads: int
    sdpa: bool
    dropout: float
    reproject: bool

    def build(self) -> nn.Module:
        return LatentMultiHeadSelfAttention(self.positional_encoding_factory,
                                                             projection_dim=self.projection_dim,
                                                             embedding_dim=self.ctx.embedding_dim,
                                                             v_dim=self.v_dim,
                                                             causal=self.causal,
                                                             n_heads=self.n_heads,
                                                             sdpa=self.sdpa,
                                                             dropout=self.dropout,
                                                             reproject=self.reproject)

# ---------- Layer Registration ---------- #

AttentionFactory = Annotated[
    Union[MultiHeadSelfAttentionFactory, LatentMultiHeadSelfAttentionFactory], Field(discriminator="type")]