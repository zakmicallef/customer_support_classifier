from typing import List
from pydantic import BaseModel

class ZeroShotClassificationResult(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]

class AiResponse(BaseModel):
    category: str
    confidence: float
    summary: str