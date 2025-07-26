from db import get_local_session
from db.dataset import get_cs_tickets_df, seed_database_from_df

def seed_database():
    sessionMaker = get_local_session()
    session = sessionMaker()
    dataset_df = get_cs_tickets_df()
    seed_database_from_df(dataset_df, session)
