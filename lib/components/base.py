from pydantic import BaseModel, ConfigDict

class BuildContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    embedding_dim: int = 256
    max_context: int = 512