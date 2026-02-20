from typing import Literal, Annotated, Union, Optional
from pydantic import BaseModel, ConfigDict, model_validator, Field

from torch.nn import Module
from torch import nn
from torch import Tensor
import torch
from functools import cache
from torch.cuda.amp import autocast

from torch.nn import *

from lib.components.base import BuildContext

@cache
def closest_square(batches, features):
    divisor = int(features ** 0.5)
    while features % divisor != 0:
        divisor -= 1

    return (batches, divisor, features // divisor)

# ---------- Layer Definitions ---------- #

class IdentityFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["identity"] = "identity"

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.Identity()


class SVDTruncation(Module):
    r"""Truncates the singular values of a reshaped vector using SVD. Truncates via singular value count or via thresholding.

    Args:
        eps (float, optional): Singular values smaller than eps will be discarded.:
            Default: ``None``
        k (int, optional): Number of singular values to keep.

    Shape:
        - Input: (..., Features)
        - Output: (..., Features)
    """
    def __init__(self, eps: float | None, k: int | None) -> None:
        super().__init__()
        if eps is None and k is None:
            raise ValueError("Need to specify either eps or k")

        self.eps = eps
        self.k = k


    def forward(self, input: Tensor) -> Tensor:
        flattened_input = input.reshape(-1, input.shape[-1])
        targ_shape = closest_square(*flattened_input.shape)

        A = flattened_input.reshape(targ_shape)
        U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
        mask = torch.ones_like(S)
        if self.k is not None:
            mask[:,self.k:] = 0
        if self.eps is not None:
            mask[S < (self.eps * S.max())] = 0

        S = S * mask
        return torch.bmm(U * S.unsqueeze(-2), Vh).reshape(input.shape).to(dtype=input.dtype)

class SVDTruncationFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["svdtruncation"] = "svdtruncation"

    eps: float | None
    k: int | None

    @model_validator(mode="after")
    def _check(self):
        if self.eps is None and self.k is None:
            raise ValueError("SVDTruncation: specify either eps or k")
        return self

    def build(self, ctx: BuildContext) -> nn.Module:
        return SVDTruncation(eps=self.eps, k=self.k)


class QRTruncation(Module):
    r"""Truncates the rank of a reshaped vector using QR.

    Args:
        k (int, optional): Rank of output.

    Shape:
        - Input: (..., Features)
        - Output: (..., Features)
    """
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        flattened_input = input.reshape(-1, input.shape[-1])
        targ_shape = closest_square(*flattened_input.shape)

        with autocast(enabled=False):
            A = flattened_input.reshape(targ_shape).float()
            omega = torch.randn(targ_shape[0], targ_shape[-1], self.k, device=A.device, dtype=A.dtype)
            Y = torch.bmm(A, omega)
            Q, _ = torch.linalg.qr(Y, mode='reduced')

            output = torch.bmm(Q, torch.bmm(Q.mT, A))

        return output.reshape(input.shape).to(dtype=input.dtype)

class QRTruncationFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["qrtruncation"] = "qrtruncation"

    k: int

    def build(self, ctx: BuildContext) -> nn.Module:
        return QRTruncation(k=self.k)


class SVDEntropicReduction(Module):
    r"""Reduces the entropy of the singular values of a reshaped vector using SVD.
    Modifies the singular values with via:
    u_i = x_i^\alpha
    x'_i = u_i * norm(x) / norm(u)

    This conserves the frobenius norm, but pushes weight from the lower end of the spectrum to the upper end,
    hence decreasing the entropy.

    Args:
        alha (float, required): Strength of the entropic reduction. must be >1:
            Default: ``None``

    Shape:
        - Input: (..., Features)
        - Output: (..., Features)
    """
    def __init__(self, alpha) -> None:
        super().__init__()
        assert alpha > 1
        self.alpha = alpha


    def forward(self, input: Tensor) -> Tensor:
        flattened_input = input.reshape(-1, input.shape[-1])
        targ_shape = closest_square(*flattened_input.shape)

        A = flattened_input.reshape(targ_shape)
        U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)

        u = torch.pow(S, self.alpha)
        S = u * torch.linalg.norm(S, dim=-1, keepdim=True) / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)


        return torch.bmm(U * S.unsqueeze(-2), Vh).reshape(input.shape).to(dtype=input.dtype)

class SVDEntropicReductionFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["svdentropicreduction"] = "svdentropicreduction"

    alpha: float

    def build(self, ctx: BuildContext) -> nn.Module:
        return SVDEntropicReduction(alpha=self.alpha)


class GELUFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["gelu"] = "gelu"

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.GELU()


class RELUFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["relu"] = "relu"

    def build(self, ctx: BuildContext) -> nn.Module:
        return nn.ReLU()

# ---------- Layer Registration ---------- #

ActivationFactory = Annotated[
        Union[
        IdentityFactory, GELUFactory, RELUFactory, QRTruncationFactory, SVDTruncationFactory, SVDEntropicReductionFactory], Field(
        discriminator="type")]
