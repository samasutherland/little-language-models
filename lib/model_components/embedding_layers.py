from typing import Literal, Annotated, Union
from pydantic import ConfigDict, Field

from torch import nn

from lib import Context, Factory


# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(Factory[nn.Module]):
    
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    def build(self, ctx: Context) -> nn.Module:
        vocab_size = ctx.require("vocab_size")
        padding_idx = ctx.require("padding_idx")
        embedding_dim = ctx.require("embedding_dim")

        return nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]