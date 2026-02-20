from pydantic import BaseModel, ConfigDict


class BuildContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    embedding_dim: int
    qk_dim: int | None = None
    max_context: int | None = None
    
    def fork(self, **updates):
        """Create a new BuildContext with updated values."""
        return self.model_copy(update=updates)
    
    def require(self, name: str):
        """Assert that a field is present and return it, raising ValueError if missing."""
        v = getattr(self, name)
        if v is None:
            raise ValueError(f"BuildContext missing required field: {name}")
        return v