from datetime import datetime
from pydantic import BaseModel, model_validator
from typing import Optional

class RequestCreate(BaseModel):
    content: str
    subject: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def allow_text_or_body(cls, values):
        values['content'] = values.get('content') or values.get('text') or values.get('body'); return values
    

class RequestResponse(BaseModel):
    id: int
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    predicted_category: Optional[str] = None
    confidence: Optional[str] = None
    summary: Optional[str] = None