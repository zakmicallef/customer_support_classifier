from datetime import datetime
from pydantic import BaseModel, computed_field, model_validator
from typing import Optional

class RequestCreate(BaseModel):
    content: str
    subject: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def allow_text_or_body(cls, values):
        values['content'] = values.get('content') or values.get('text') or values.get('body'); return values
    
    @computed_field(return_type=str)
    @property
    def get_info(self) -> str:
        subject = self.subject
        if subject:
            return f"{subject}, {self.content}"

        return self.content

class RequestResponse(BaseModel):
    id: int
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    predicted_category: Optional[str] = None
    confidence: Optional[str] = None
    summary: Optional[str] = None


class CategoryQueryParams(BaseModel):
    category: Optional[str] = None