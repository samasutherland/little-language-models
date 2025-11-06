import torch
from torch import nn
import torch.nn.functional as F
from .activations import SVDEntropicReduction, SVDTruncation, QRTruncation

class Transformer(nn.Module):
    def __init__(self, model_kwargs):
        super().__init__()

        global_kwargs = model_kwargs["global"]
        attention_kwargs = model_kwargs["attention"]
        norm_kwargs = model_kwargs["norm"]
        dropout_kwargs = model_kwargs["dropout"]
        rope_kwargs = model_kwargs["rope"]
        activation_kwargs = model_kwargs["activation_kwargs"]

        norm = NORM_REGISTRY[norm_kwargs["type"]]
        updated_norm_kwargs = {key: val for key, val in norm_kwargs.items() if key != "type"}


        self.embedding = nn.Embedding(global_kwargs["vocab_size"], global_kwargs["embedding_dim"], padding_idx=global_kwargs.get("padding_idx", 3))
        self.transformer_stack = nn.Sequential(*[TransformerLayer(global_kwargs, attention_kwargs, dropout_kwargs, updated_norm_kwargs, rope_kwargs, activation_kwargs, norm) for _ in range(global_kwargs["num_layers"])])


        self.final_norm = norm(global_kwargs["embedding_dim"], **{} if updated_norm_kwargs is None else updated_norm_kwargs)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.t())
        return logits