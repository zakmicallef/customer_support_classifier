import math
from models.rag import Rag
from processes.preprocess import get_cs_tickets_df
from util.load_configs import load_data_config


def ingest():
    config = load_data_config()
    rag = Rag(chroma_file_name=config.rag_save_file_name, persistent=True)
    rag.ingest()