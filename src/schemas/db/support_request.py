from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from schemas.db import Base

class SupportRequest(Base):
    __tablename__ = 'support_requests'

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, nullable=True)
    body = Column(String, nullable=True)
    resolved = Column(Boolean, default=False)
    predicted_category = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    summary = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(), nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    status = Column(String)