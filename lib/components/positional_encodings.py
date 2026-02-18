import torch
from torch import nn
from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, TypeAdapter

from base import BuildContext

# ---------- Layer Definitions ---------- #

class RoPE(nn.Module):
    def __init__(self, max_seq_len, dim, base=10000):
        super().__init__()
        m_values = torch.arange(max_seq_len, requires_grad=False)
        assert dim % 2 == 0, "RoPE requires even dimension"

        theta_values = torch.pow(base, -2 * torch.arange(dim//2, requires_grad=False) / dim)
        trig_args = torch.outer(m_values, theta_values) # max_seq_len, dim
        # I need to wrap these in a buffer or parameter so that they get moved to the correct device when the model is moved.
        cos_vals = torch.cos(trig_args)
        sin_vals = torch.sin(trig_args)# multiply by alternation -1 +1s for correct rotation application
        self.register_buffer("cos_vals", cos_vals, persistent=False)
        self.register_buffer("sin_vals", sin_vals, persistent=False)

    def forward(self, x):
        batch_dim, n_heads, seq, qk_dim = x.shape
        # x is batch, n_heads, seq, qk_dim
        x1, x2 = x[...,::2], x[...,1::2]
        cos = self.cos_vals[:seq].to(dtype=x.dtype, device=x.device)
        sin = self.sin_vals[:seq].to(dtype=x.dtype, device=x.device)
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).contiguous()

        return x_rotated


class RoPEFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["rope"] = "rope"
    base: int = 10000

    ctx: BuildContext = BuildContext()
    qk_dim: int = 64

    def build(self) -> nn.Module:
        return RoPE(self.ctx.max_context, self.qk_dim, base=self.base)

# ---------- Layer Registration ---------- #

PositionalEncodingFactory = Annotated[Union[RoPEFactory], Field(discriminator="type")]

