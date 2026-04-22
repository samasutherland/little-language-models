from typing import Literal, Annotated, Union
from sentencepiece import SentencePieceProcessor

from pydantic import ConfigDict, Field

from lib import Context, Factory



class SentencePieceFactory(Factory[SentencePieceProcessor]):
    
    type: Literal["sentencepiece"] = "sentencepiece"

    tokenizer_path: str

    def build(self, ctx: Context) -> SentencePieceProcessor:
        tokenizer_path = getattr(ctx, "tokenizer_path", self.tokenizer_path)
        if "{vocab_size}" in tokenizer_path:
            if not hasattr(ctx, "vocab_size"):
                raise ValueError("Context missing required field: vocab_size for tokenizer_path template.")
            vocab_size = getattr(ctx, "vocab_size")
            tokenizer_path = tokenizer_path.format(vocab_size=vocab_size)
        return SentencePieceProcessor(model_file=tokenizer_path)


TokenizerFactory = Annotated[
    Union[SentencePieceFactory], Field(discriminator="type")]