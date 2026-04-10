from typing import Literal, Annotated, Union
from pydantic import ConfigDict, Field

from torch import nn

from lib import Context, Factory


# ---------- Layer Definitions ---------- #

class StandardEmbeddingLayerFactory(Factory[nn.Module]):
    
    type: Literal["standardembeddinglayer"] = "standardembeddinglayer"

    def build(self, ctx: Context) -> nn.Module:
        # vocab_size = ctx.require("vocab_size")
        # pad_id = ctx.require("pad_id")
        train_dataset = ctx.require("train_dataloader").dataset
        embedding_dim = ctx.require("embedding_dim")
        
        pad_id = train_dataset.pad_id
        vocab_size = train_dataset.vocab_size

        return nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

# ---------- Layer Registration ---------- #

EmbeddingLayerFactory = Annotated[
    Union[StandardEmbeddingLayerFactory], Field(discriminator="type")]