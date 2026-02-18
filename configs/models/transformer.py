from lib.components import *

ctx = BuildContext(embedding_dim=256)

norm_options = {"type": "rmsnorm",
                "ctx": ctx}
norm_factory = RMSNormFactory(**norm_options)

activation_options = {"type": "gelu",
                      "ctx": ctx}
activation_factory = GELUFactory(**activation_options)

positional_encoding_options = {"type": "rope",
                               "ctx": ctx,
                               "base": 10000,
                               "max_context": 512,
                               "qk_dim": 64}
positional_encoding_factory = RoPEFactory(**positional_encoding_options)

attention_options = {"type": "multiheadselfattention",
                     "ctx": ctx,
                     "positional_encoding_factory": positional_encoding_factory,
                     "v_dim": 64,
                     "causal": True,
                     "n_heads": 4,
                     "sdpa": True,
                     "dropout": 0.0,
                     "reproject": True}
attention_factory = LatentMultiHeadSelfAttentionFactory(**attention_options)

transformer_layer_options = {"type": "standardtransformerlayer",
                             "ctx": ctx,
                             "activation_factory": activation_factory,
                             "norm_factory": norm_factory,
                             "attention_factory": attention_factory,
                             "dropout": 0.0,
                             "feedforward_dim": 1024}
transformer_layer_factory = TransformerLayerFactory(**transformer_layer_options)

final_norm_options = {"type": "rmsnorm",
                      "ctx": ctx}
final_norm_factory = RMSNormFactory(**final_norm_options)

embedding_options = {"type": "standardembeddinglayer",
                     "ctx": ctx,
                     "vocab_size": 4000,
                     "padding_idx": 3}
embedding_layer_factory = EmbeddingLayerFactory(**embedding_options)

language_model_options = {"type": "transformer",
                          "ctx": ctx,
                          "transformer_layer_factory": transformer_layer_factory,
                          "final_norm_factory": final_norm_factory,
                          "embedding_layer_factory": embedding_layer_factory,
                          "num_layers": 2}
transformer_factory = TransformerFactory(**language_model_options)

model = transformer_factory.build()