class TransformerLayer(nn.Module):
    def __init__(self, global_kwargs, attention_kwargs, dropout_kwargs, norm_kwargs, rope_kwargs, activation_kwargs, norm):
        super().__init__()

        self.attn_norm = norm(global_kwargs["embedding_dim"], **norm_kwargs)
        self.ffn_norm = norm(global_kwargs["embedding_dim"], **norm_kwargs)

        self.attn_dropout = nn.Dropout(dropout_kwargs["attn_dropout"])
        self.ffn_dropout = nn.Dropout(dropout_kwargs["ffn_dropout"])

        self.attention = MaskedMultiHeadLatentRoPESelfAttention(embedding_dim=global_kwargs["embedding_dim"], max_context=global_kwargs["max_context"], dropout=dropout_kwargs["attn_dropout"], rope_kwargs=rope_kwargs, **attention_kwargs)

        if global_kwargs["activation"] == "Identity":
            self.ffn = nn.Identity() # Identity activation collapses ff layers to no-op.
        else:
            self.ffn = nn.Sequential(nn.Linear(global_kwargs["embedding_dim"], global_kwargs["feedforward_dim"]),
                                     ACTIVATION_REGISTRY[global_kwargs["activation"]](**activation_kwargs),
                                     self.ffn_dropout,
                                     nn.Linear(global_kwargs["feedforward_dim"], global_kwargs["embedding_dim"]))


    def forward(self, x):
        x = x[:, -self.attention.max_context:, :]
        x = x + self.attn_dropout(self.attention(self.attn_norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x
