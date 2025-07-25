from typing import Optional
from pydantic import BaseModel

class LiveConfig(BaseModel):
    priority_model_name: str
    queue_model_name: str
    priority_stop: bool
    rag_store_name: Optional[str]

class TestConfig(BaseModel):
    priority_model_name: str
    queue_model_name: str
    test_factor: float
    rag_store_name: Optional[str]
