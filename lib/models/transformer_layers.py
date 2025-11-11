from torch import nn
from lib.models import attention_layers
from lib.models import activations


class TransformerLayer(nn.Module):
    def __init__(self, norm, attention, ):
        super().__init__()

        self.norm = norm

        self.attn_dropout = nn.Dropout(dropout_kwargs["attn_dropout"])
        self.ffn_dropout = nn.Dropout(dropout_kwargs["ffn_dropout"])

        self.attention = attention

        if global_kwargs["activation"] == "Identity":
            self.ffn = nn.Identity() # Identity activation collapses ff layers to no-op.
        else:
            self.ffn = nn.Sequential(nn.Linear(global_kwargs["embedding_dim"], global_kwargs["feedforward_dim"]),
                                     getattr(activations, global_kwargs["activation"])(**activation_kwargs),
                                     self.ffn_dropout,
                                     nn.Linear(global_kwargs["feedforward_dim"], global_kwargs["embedding_dim"]))


    def forward(self, x):
        x = x[:, -self.attention.max_context:, :]
        x = x + self.attn_dropout(self.attention(self.attn_norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x
