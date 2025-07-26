from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Table, DateTime
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

from schemas.db import Base

# Association table for many-to-many tags
ticket_tags = Table(
    'ticket_tags',
    Base.metadata,
    Column('ticket_id', UUID(as_uuid=True), ForeignKey('tickets.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class Priority(Base):
    __tablename__ = 'priorities'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    tickets = relationship("Ticket", back_populates="priority")

class Queue(Base):
    __tablename__ = 'queues'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    tickets = relationship("Ticket", back_populates="queue")

class Language(Base):
    __tablename__ = 'languages'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    tickets = relationship("Ticket", back_populates="language")

class Tag(Base):
    __tablename__ = 'tags'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    tickets = relationship("Ticket", secondary=ticket_tags, back_populates="tags")

class Ticket(Base):
    __tablename__ = 'tickets'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(Text)
    body = Column(Text)
    answer = Column(Text)
    type = Column(String)

    priority_id = Column(Integer, ForeignKey('priorities.id'))
    queue_id = Column(Integer, ForeignKey('queues.id'))
    language_id = Column(Integer, ForeignKey('languages.id'))

    priority = relationship("Priority", back_populates="tickets")
    queue = relationship("Queue", back_populates="tickets")
    language = relationship("Language", back_populates="tickets")

    tags = relationship("Tag", secondary=ticket_tags, back_populates="tickets")
