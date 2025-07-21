from schemas.pydantic.postgres_config import PostgresSettings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine():
    return create_engine(PostgresSettings().db_url)

def get_local_session():
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)