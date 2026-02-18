from lib.components import *

embedding_dim = 256

norm_options = {
    "type": "rmsnorm",
    "embedding_dim": embedding_dim,
}
norm_factory = RMSNormFactory(**norm_options)

activation_options = {
    "type": "gelu",
}
activation_factory = GELUFactory(**activation_options)

positional_encoding_options = {
    "type": "rope",
    "base": 10000,
    "max_context": 512,
    "qk_dim": 64,
}
positional_encoding_factory = RoPEFactory(**positional_encoding_options)

attention_options = {
    "type": "multiheadselfattention",
    "embedding_dim": embedding_dim,
    "positional_encoding_factory": positional_encoding_factory,
    "qk_dim": 64,
    "max_context": 512,
    "v_dim": 64,
    "causal": True,
    "n_heads": 4,
    "sdpa": True,
    "dropout": 0.0,
    "reproject": True,
}
attention_factory = LatentMultiHeadSelfAttentionFactory(**attention_options)

transformer_layer_options = {
    "type": "standardtransformerlayer",
    "activation_factory": activation_factory,
    "norm_factory": norm_factory,
    "attention_factory": attention_factory,
    "embedding_dim": embedding_dim,
    "dropout": 0.0,
    "feedforward_dim": 1024,
}
transformer_layer_factory = TransformerLayerFactory(**transformer_layer_options)

final_norm_options = {
    "type": "rmsnorm",
    "embedding_dim": embedding_dim,
}
final_norm_factory = RMSNormFactory(**final_norm_options)

embedding_options = {
    "type": "standardembeddinglayer",
    "embedding_dim": embedding_dim,
    "vocab_size": 4000,
    "padding_idx": 3,
}
embedding_layer_factory = EmbeddingLayerFactory(**embedding_options)

language_model_options = {
    "type": "transformer",
    "embedding_dim": embedding_dim,
    "transformer_layer_factory": transformer_layer_factory,
    "final_norm_factory": final_norm_factory,
    "embedding_layer_factory": embedding_layer_factory,
    "num_layers": 2,
}
transformer_factory = TransformerFactory(**language_model_options)

model = transformer_factory.build()