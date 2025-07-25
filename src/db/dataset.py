# TODO Move to a better place
from processes.preprocess import get_cs_tickets_df
from util.load_configs import load_data_config


from loguru import logger


import math


def get_dataset():
    config = load_data_config()
    cs_tickets_df = get_cs_tickets_df()

    len_of_dataset = cs_tickets_df.shape[0]
    cs_tickets_df = cs_tickets_df[:-math.floor(len_of_dataset*config.test_factor)]

    texts = cs_tickets_df['body']

    logger.info(f'Data loaded for testing')
    return cs_tickets_df, texts