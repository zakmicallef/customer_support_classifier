from db import get_engine

from schemas.db import *
from schemas.db.support_request import *

def init_tables():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)