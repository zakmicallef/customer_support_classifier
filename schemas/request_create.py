from pydantic import BaseModel
from typing import Optional

class RequestCreate(BaseModel):
    text: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None