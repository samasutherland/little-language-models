import torch
from torch import nn
import torch.nn.functional as F

NORM_REGISTRY = {"RMSNorm": nn.RMSNorm}
ACTIVATION_REGISTRY = {"GELU": nn.GELU}


class RoPETransform(nn.Module):
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

class MaskedMultiHeadLatentRoPESelfAttention(nn.Module):
    def __init__(self, embedding_dim=384, project_kv=True, projection_dim=128, qk_dim=64, v_dim=64, causal=True, max_context=100, n_heads=6, sdpa=False, dropout=0.1, rope_kwargs=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.projection_dim = projection_dim
        self.max_context = max_context
        self.causal = causal
        self.n_heads = n_heads
        self.sdpa = sdpa
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)


        self.inv_sqrt_qk_dim = 1/qk_dim ** 0.5

        if rope_kwargs is None:
            rope_kwargs = {}

        self.qk_rope = RoPETransform(max_context, qk_dim, **rope_kwargs)
        #self.v_rope = RoPETransform(max_context, v_dim)

        # Want the total number of parameters for kv to be = to the number of parameters for q.
        # q_params = d_embed * d_qk * n_heads
        # kv_params = d_embed * d_proj + d_proj * (d_qk + d_v) * n_heads
        # In the previous layers, I speparated d_qk and dv, letting n_heads * d_v = d_embed. This means that the concatenation of all heads injects a full embedding's worth of information
        # After reading a bit more, it might make sense for d_v = d_qk = d_embed / n_heads. This means that each attention head attends to 1/n_heads worth of embedding, then also injects ~ 1/n_heads embeddings worth of information. Nice and neat. Clarification needed on the empirically optimal choice of these dimensions, as well as the relative balance of parameter counts between q, k, and v.

        # Taking d_v = d_qk, and setting q_params = kv_params, we get:
        # d_embed * d_qk * n_heads = d_embed * d_proj + d_proj * 2 * d_qk * n_heads
        # d_proj = d_embed * d_qk * n_heads / (d_embed + 2 * d_qk * n_heads)
        # if d_qk = d_embed / n_heads:
        # d_proj = d_embed / 3

        # For the default parameters, let embedding_dim = 384, n_heads = 6
        # d_qk = 384 / 6 = 64
        # d_v = d_qk = 384 / 6 = 64
        # d_proj = 384 / 3 = 128

        self.q = nn.Linear(embedding_dim, qk_dim * n_heads) # Grouped matrix for the heads
        if project_kv:
            self.kv = nn.Sequential(nn.Linear(embedding_dim, projection_dim), nn.Linear(projection_dim, (qk_dim + v_dim) * n_heads)) # Grouped matrix for the heads and k and v
        else:
            self.kv = nn.Linear(embedding_dim, (qk_dim + v_dim) * n_heads)

        if causal:
            mask = torch.zeros(max_context, max_context)
            self.register_buffer("mask_array", mask.masked_fill(~torch.tril(torch.ones(max_context, max_context, dtype=torch.bool)), float("-inf")))

        self.reproject = nn.Linear(n_heads * v_dim, embedding_dim)

    def forward(self, x):

        x = x[:, -self.max_context:, :]
        batch_dim, seq_len, embed_dim = x.shape
        q = self.q(x) # batch, seq, n_heads * qk_dim

        kv = self.kv(x) # batch, seq, n_heads * (qk_dim + v_dim)
        k, v = torch.split(kv, [self.qk_dim * self.n_heads, self.v_dim * self.n_heads], dim=-1) # batch, seq, n_heads * qk_dim; batch, seq, n_heads * v_dim

        # Use view to prevent copying
        q = self.qk_rope(q.view(batch_dim, seq_len, self.n_heads, self.qk_dim).transpose(1, 2))  # batch, n_heads, seq, qk_dim
        k = self.qk_rope(k.view(batch_dim, seq_len, self.n_heads, self.qk_dim).transpose(1,2)) # batch, n_heads, seq, qk_dim
        v = v.view(batch_dim, seq_len, self.n_heads, self.v_dim).transpose(1,2) #self.v_rope(v.view(batch_dim, seq_len, self.n_heads, self.v_dim).transpose(1,2)) # batch, n_heads, seq, v_dim

        if self.sdpa:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.dropout)
        else:

            attn_scores = q @ k.transpose(2,3) * self.inv_sqrt_qk_dim  # batch, n_heads, seq, seq

            if self.causal:
                attn_scores += self.mask_array[-seq_len:, -seq_len:]

            attn_probs = torch.softmax(attn_scores, dim=-1)  # batch, n_heads, seq, seq
            if self.dropout > 0 and self.training:
                attn_probs = self.dropout_layer(attn_probs)
            attn_output = attn_probs @ v  # batch, n_heads, seq, v_dim

        #TODO: this reshape copies data. view would be better if I can get the shapes to work.
        return self.reproject(attn_output.transpose(1, 2).reshape(batch_dim, seq_len,
                                                         self.v_dim * self.n_heads))  # batch, seq, embedding_dim

class TransformerLayer(nn.Module):
    def __init__(self, global_kwargs, attention_kwargs, dropout_kwargs, norm_kwargs, rope_kwargs, norm):
        super().__init__()

        self.attn_norm = norm(global_kwargs["embedding_dim"], **norm_kwargs)
        self.ffn_norm = norm(global_kwargs["embedding_dim"], **norm_kwargs)

        self.attn_dropout = nn.Dropout(dropout_kwargs["attn_dropout"])
        self.ffn_dropout = nn.Dropout(dropout_kwargs["ffn_dropout"])

        self.attention = MaskedMultiHeadLatentRoPESelfAttention(embedding_dim=global_kwargs["embedding_dim"], max_context=global_kwargs["max_context"], dropout=dropout_kwargs["attn_dropout"], rope_kwargs=rope_kwargs, **attention_kwargs)

        self.ffn = nn.Sequential(nn.Linear(global_kwargs["embedding_dim"], global_kwargs["feedforward_dim"]), ACTIVATION_REGISTRY[global_kwargs["activation"]](), self.ffn_dropout, nn.Linear(global_kwargs["feedforward_dim"], global_kwargs["embedding_dim"]))


    def forward(self, x):
        x = x[:, -self.attention.max_context:, :]
        x = x + self.attn_dropout(self.attention(self.attn_norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, model_kwargs):
        super().__init__()

        global_kwargs = model_kwargs["global"]
        attention_kwargs = model_kwargs["attention"]
        norm_kwargs = model_kwargs["norm"]
        dropout_kwargs = model_kwargs["dropout"]
        rope_kwargs = model_kwargs["rope"]

        norm = NORM_REGISTRY[norm_kwargs["type"]]
        updated_norm_kwargs = {key: val for key, val in norm_kwargs.items() if key != "type"}


        self.embedding = nn.Embedding(global_kwargs["vocab_size"], global_kwargs["embedding_dim"])
        self.transformer_stack = nn.Sequential(*[TransformerLayer(global_kwargs, attention_kwargs, dropout_kwargs, updated_norm_kwargs, rope_kwargs, norm) for _ in range(global_kwargs["num_layers"])])


        self.final_norm = norm(global_kwargs["embedding_dim"], **{} if updated_norm_kwargs is None else updated_norm_kwargs)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_stack(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.t())
        return logits