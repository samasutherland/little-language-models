import torch
from torch import nn
import torch.nn.functional as F
from .activations import SVDEntropicReduction, SVDTruncation, QRTruncation
from lib.models import transformer_layers

class Transformer(nn.Module):
    def __init__(self, model_kwargs):
        super().__init__()

        global_kwargs = model_kwargs["global"]
        attention_kwargs = model_kwargs["attention_kwargs"]
        norm_kwargs = model_kwargs["norm_kwargs"]
        dropout_kwargs = model_kwargs["dropout"]
        positional_encoding_kwargs = model_kwargs["positional_encoding_kwargs"]
        activation_kwargs = model_kwargs["activation_kwargs"]

        self.embedding = nn.Embedding(global_kwargs["vocab_size"], global_kwargs["embedding_dim"], padding_idx=global_kwargs.get("padding_idx", 3))
        self.transformer_stack = nn.Sequential(*[getattr(transformer_layers, global_kwargs["transformer_layer"])(global_kwargs, attention_kwargs, dropout_kwargs, norm_kwargs, positional_encoding_kwargs, activation_kwargs) for _ in range(global_kwargs["num_layers"])])


        self.final_norm = getattr(nn, global_kwargs["norm"])(global_kwargs["embedding_dim"], **norm_kwargs)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.t())
        return logits