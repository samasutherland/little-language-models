from pydantic import BaseModel, ConfigDict

class BuildContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    embedding_dim: int = 256