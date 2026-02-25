from typing import Optional, Literal, Annotated, Union
from sentencepiece import SentencePieceProcessor

from pydantic import BaseModel, ConfigDict, Field

from lib.data_components.context import DataContext



class SentencePieceFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sentencepiece"] = "sentencepiece"

    tokenizer_path: str

    def build(self, ctx: DataContext) -> SentencePieceProcessor:
        return SentencePieceProcessor(model_file=self.tokenizer_path)


TokenizerFactory = Annotated[
    Union[SentencePieceFactory], Field(discriminator="type")]