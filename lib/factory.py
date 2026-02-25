from typing import Literal, Annotated, Union
from pydantic import BaseModel, ConfigDict, Field


class Factory(BaseModel):
    @overload