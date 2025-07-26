from typing import List, Optional
from pydantic import BaseModel

class ZeroShotClassificationResult(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]

class AiResponse(BaseModel):
    category: str
    confidence: Optional[float] = None
    summary: Optional[str] = None