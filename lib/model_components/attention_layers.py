import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field

from lib.model_components.base import BuildContext
from lib.model_components.positional_encodings import PositionalEncodingFactory

# ---------- Layer Definitions ---------- #

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 positional_encoding: nn.Module,
                 embedding_dim: int,
                 qk_dim: int,
                 max_context: int,
                 v_dim: int,
                 causal: bool,
                 n_heads: int,
                 sdpa: bool,
                 dropout: float,
                 reproject: bool):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.qk_dim = qk_dim
        self.max_context = max_context
        self.v_dim = v_dim
        self.causal = causal
        self.n_heads = n_heads
        self.sdpa = sdpa
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.qk_positional_encoding = positional_encoding

        self.inv_sqrt_qk_dim = 1 / self.qk_dim ** 0.5

        self.q = nn.Linear(embedding_dim, self.qk_dim * n_heads)  # Grouped matrix for the heads
        self.kv = nn.Linear(embedding_dim, (self.qk_dim + v_dim) * n_heads)

        if causal:
            mask = torch.zeros(self.max_context, self.max_context)
            self.register_buffer("mask_array",
                                 mask.masked_fill(~torch.tril(torch.ones(self.max_context, self.max_context, dtype=torch.bool)),
                                                  float("-inf")))

        if n_heads * v_dim != embedding_dim or reproject:
            self.reproject = nn.Linear(n_heads * v_dim, embedding_dim)
        else:
            self.reproject = nn.Identity()

    def forward(self, x):

        x = x[:, -self.max_context:, :]
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
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.dropout if (self.dropout > 0 and self.training) else 0.0)
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

    max_context: int
    qk_dim: int
    v_dim: int
    causal: bool
    n_heads: int
    sdpa: bool
    dropout: float
    reproject: bool

    def build(self, ctx: BuildContext) -> nn.Module:
        ctx_fork = ctx.fork(qk_dim=self.qk_dim, max_context=self.max_context)
        positional_encoding = self.positional_encoding_factory.build(ctx_fork)

        embedding_dim = ctx.require("embedding_dim")
        
        return MultiHeadSelfAttention(
            positional_encoding,
            embedding_dim=embedding_dim,
            qk_dim=self.qk_dim,
            max_context=self.max_context,
            v_dim=self.v_dim,
            causal=self.causal,
            n_heads=self.n_heads,
            sdpa=self.sdpa,
            dropout=self.dropout,
            reproject=self.reproject,
        )


class LatentMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self,
                 positional_encoding: nn.Module,
                 embedding_dim: int,
                 qk_dim: int,
                 max_context: int,
                 projection_dim: int,
                 v_dim: int,
                 causal: bool,
                 n_heads: int,
                 sdpa: bool,
                 dropout: float,
                 reproject: bool):
        super().__init__(
            positional_encoding,
            embedding_dim=embedding_dim,
            qk_dim=qk_dim,
            max_context=max_context,
            v_dim=v_dim,
            causal=causal,
            n_heads=n_heads,
            sdpa=sdpa,
            dropout=dropout,
            reproject=reproject,
        )

        # Overwrite kv transform with latent space version
        self.kv = nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.Linear(projection_dim, (
                    self.qk_dim + v_dim) * n_heads))  # Grouped matrix for the heads and k and v

class LatentMultiHeadSelfAttentionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["latentmultiheadselfattention"] = "latentmultiheadselfattention"

    positional_encoding_factory: PositionalEncodingFactory

    max_context: int
    qk_dim: int
    projection_dim: int
    v_dim: int
    causal: bool
    n_heads: int
    sdpa: bool
    dropout: float
    reproject: bool

    def build(self, ctx: BuildContext) -> nn.Module:
        ctx_fork = ctx.fork(qk_dim=self.qk_dim, max_context=self.max_context)
        positional_encoding = self.positional_encoding_factory.build(ctx_fork)

        embedding_dim = ctx.require("embedding_dim")
        
        return LatentMultiHeadSelfAttention(
            positional_encoding,
            embedding_dim=embedding_dim,
            qk_dim=self.qk_dim,
            max_context=self.max_context,
            projection_dim=self.projection_dim,
            v_dim=self.v_dim,
            causal=self.causal,
            n_heads=self.n_heads,
            sdpa=self.sdpa,
            dropout=self.dropout,
            reproject=self.reproject,
        )

# ---------- Layer Registration ---------- #

AttentionFactory = Annotated[
    Union[MultiHeadSelfAttentionFactory, LatentMultiHeadSelfAttentionFactory], Field(discriminator="type")]
