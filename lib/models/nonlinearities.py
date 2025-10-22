from torch.nn import Module
from torch import Tensor
import torch

class SVDTruncation(Module):
    r"""Truncates the singular values of a reshaped vector using SVD. Truncates via singular value count or via thresholding.

    Args:
        eps (float, optional): Singular values smaller than eps will be discarded.:
            Default: ``None``
        k (int, optional): Number of singular values to keep.

    Shape:
        - Input: (Batches, Features)
        - Output: (Batches, Features)
    """
    def __init__(self, eps=None, k=None) -> None:
        super().__init__()
        if eps is None and k is None:
            raise ValueError("Need to specify either eps or k")

        self.eps = eps
        self.k = k
        self.targ_shape = None

    def get_targ_shape(self, input_shape):
        batches, features = input_shape
        divisor = torch.floor(torch.sqrt(features))
        while features % divisor != 0:
            divisor -= 1

        self.targ_shape = (batches, divisor, features//divisor)

    def forward(self, input: Tensor) -> Tensor:
        if self.targ_shape is None:
            self.get_targ_shape(input.shape)

        A = input.reshape(self.targ_shape)
        U, S, V = torch.linalg.svd(A)
        mask = torch.ones(S.shape)
        if self.k is not None:
            mask[:,self.k:] = 0
        if self.eps is not None:
            mask[S < self.eps] = 0

        S = S * mask
        return torch.bmm(U, torch.bmm(torch.diag_embed(S), V.T)).reshape(input.shape)


# class EntropicReduction(Module):
#
#     def __init__(self) -> None:
#         super().__init__()
#
#
#     def forward(self, input: Tensor) -> Tensor: