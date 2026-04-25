import torch
from torch import nn
from torch.nn import ParameterList

from sympy import factorint
import warnings
from torch.nn.parameter import Parameter

try:
    from cuquantum.tensornet import contract, contract_path
    def make_optimize(path):
        return {"path": path}
except ImportError:
    warnings.warn("cuquantum not available. Falling back to opt_einsum, which may be significantly slower.")
    from opt_einsum import contract, contract_path
    def make_optimize(path):
        return path

from opt_einsum import get_symbol


def generate_symbol():
    i = 0
    while True:
        yield get_symbol(i)
        i += 1

class TensorRingLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bond_dim: int, bias: bool = True, device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dim = bond_dim
        
        in_factors = factorint(in_features)
        self.in_factors = sum([[int(key)]*int(val) for key, val in in_factors.items()], start=[])
        out_factors = factorint(out_features)
        self.out_factors = sum([[int(key)] * int(val) for key, val in out_factors.items()], start=[])

        n = max(len(self.in_factors), len(self.out_factors))
        self.in_factors += [1] * (n - len(self.in_factors))
        self.out_factors += [1] * (n - len(self.out_factors))
        self.mpo_length = n
        
        mpo_tensors = []
        for in_shape, out_shape in zip(self.in_factors, self.out_factors):
            # in lrud shape - left bond, right bond, up (input) leg, down (output) leg
            mpo_tensors.append(Parameter(torch.empty(bond_dim, bond_dim, in_shape, out_shape, **factory_kwargs)))
            
        self.mpo_tensors = ParameterList(mpo_tensors)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        self.contraction_path = None
        self.cached_ndim = None
        
        self.reset_parameters()

    def reset_parameters(self):
        n = self.mpo_length
        # Scaling sigma in this way approximates Kaiming initialisation as bond dim -> inf
        sigma = 1.0 / (
                self.bond_dim ** 0.5 *
                (self.in_features * self.bond_dim) ** (1 / (2 * n))
        )
        for tensor in self.mpo_tensors:
            nn.init.normal_(tensor, mean=0, std=sigma)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.contraction_path is None or input.ndim != self.cached_ndim:
            self.cached_ndim = input.ndim
            self.generate_contraction_expr(input)

        output = contract(
            self.expr,
            input.reshape(*input.shape[:-1], *self.in_factors),
            *self.mpo_tensors,
            optimize=make_optimize(self.contraction_path)
        )
        output = output.reshape(*input.shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def generate_contraction_expr(self, input):
        symbols = iter(generate_symbol())
        
        bond_symbols = [next(symbols) for _ in range(self.mpo_length)]
        input_symbols = [next(symbols) for _ in range(self.mpo_length)]
        output_symbols = [next(symbols) for _ in range(self.mpo_length)]
        
        mpo_shapes = [f"{bond_symbols[i%self.mpo_length]}{bond_symbols[(i+1)%self.mpo_length]}{input_symbols[i]}{output_symbols[i]}" for i in range(self.mpo_length)]

        extra_dims = input.ndim - 1
        batch_symbols = [next(symbols) for _ in range(extra_dims)]

        input_shape = ''.join(batch_symbols + input_symbols)
        output_shape = ''.join(batch_symbols + output_symbols)

        expr = ', '.join([input_shape] + mpo_shapes) + '->' + output_shape
        contraction_path, _ = contract_path(expr, input.reshape(*input.shape[:-1], *self.in_factors), *self.mpo_tensors)
        
        self.expr = expr
        self.contraction_path = contraction_path
        
