from db import get_session_maker
from db.dataset import seed_database_from_df
from util.load_dataset import get_cs_tickets_df

def seed_database():
    sessionMaker = get_session_maker()
    session = sessionMaker()
    dataset_df = get_cs_tickets_df()
    seed_database_from_df(dataset_df, session)
