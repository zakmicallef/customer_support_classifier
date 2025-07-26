from schemas.pydantic.postgres_config import PostgresSettings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine():
    return create_engine(PostgresSettings().db_url)

def get_session_maker():
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session_maker_and_engine():
    engine = get_engine()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session, engine