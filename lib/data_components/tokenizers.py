from typing import Literal, Annotated, Union
from sentencepiece import SentencePieceProcessor

from pydantic import ConfigDict, Field

from lib import Context, Factory



class SentencePieceFactory(Factory[SentencePieceProcessor]):
    
    type: Literal["sentencepiece"] = "sentencepiece"

    tokenizer_path: str

    def build(self, ctx: Context) -> SentencePieceProcessor:
        return SentencePieceProcessor(model_file=self.tokenizer_path)


TokenizerFactory = Annotated[
    Union[SentencePieceFactory], Field(discriminator="type")]