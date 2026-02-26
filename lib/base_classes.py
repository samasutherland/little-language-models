from pydantic import BaseModel, ConfigDict

from typing import Generic, TypeVar

class Context(BaseModel):
    model_config = ConfigDict(extra="allow")

    def fork(self, **updates):
        """Create a new Context with updated values."""
        return self.model_copy(update=updates)

    def require(self, name: str):
        """Assert that a field is present and return it, raising ValueError if missing."""
        v = getattr(self, name)
        if v is None:
            raise ValueError(f"Context missing required field: {name}")
        return v

T = TypeVar("T")   # component type

class Factory(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    def build(self, ctx: Context) -> T:
        raise NotImplementedError()