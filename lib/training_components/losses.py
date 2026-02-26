from typing import Literal, Annotated, Union
from pydantic import Field

from lib import Context, Factory

from torch import nn

class CrossEntropyLossFactory(Factory[nn.Module]):
    type: Literal["crossentropy"] = "crossentropy"

    def build(self, ctx: Context) -> nn.Module:
        pad_id = ctx.require("pad_id")
        return nn.CrossEntropyLoss(ignore_index=pad_id)


LossFactory = Annotated[
    Union[CrossEntropyLossFactory],
    Field(discriminator="type"),
]

