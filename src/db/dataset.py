# TODO Move to a better place
import pandas as pd
from util.load_configs import load_data_config
from loguru import logger
import math


def apply_parse_data_points(df):
    '''
    This adds the data point needed to implement models 
    '''

    TECHNICAL = 'technical'
    BILLING = 'billing'
    GENERAL = 'general'

    # map queue
    queue_map = {
        'Technical Support': TECHNICAL,
        'IT Support': TECHNICAL,
        'Billing and Payments': BILLING,
        'Customer Service': GENERAL,
        'Product Support': GENERAL,
        'Service Outages and Maintenance': GENERAL,
        'Human Resources': GENERAL,
        'Sales and Pre-Sales': GENERAL,
        'Returns and Exchanges': GENERAL,
        'General Inquiry': GENERAL
    }
    df['queue'] = df['queue'].map(queue_map)

    if df['queue'].isna().any():
        raise AssertionError('No Nan allowed in "target"')

    return df


def apply_clean_cs_tickets(df):
    '''
    This removes unwanted data to keep the dataset clean
    '''

    # dropping german rows
    german = df['language'] == 'de'

    df = df[~german]

    # removing unwanted cols
    columns = df.columns.tolist()
    columns_keep = ['subject', 'body', 'priority', 'queue']

    drop_columns = [c for c in columns if c not in columns_keep]

    df = df.drop(columns=drop_columns)

    # dropping rows that has an empty 
    df = df.dropna(how='any')
    return df


def get_cs_tickets_df(clean_data=True, parse_data_points=True):
    df = pd.read_csv("hf://datasets/Tobi-Bueck/customer-support-tickets/dataset-tickets-multi-lang-4-20k.csv")

    if clean_data:
        df = apply_clean_cs_tickets(df)

    if parse_data_points:
        df = apply_parse_data_points(df)

    df.reset_index(inplace=True, drop=True)

    return df

def get_dataset(train=False):
    config = load_data_config()
    cs_tickets_df = get_cs_tickets_df()

    len_of_dataset = cs_tickets_df.shape[0]

    if train:
        cs_tickets_df = cs_tickets_df[:-math.floor(len_of_dataset*config.test_factor)]
    else:
        cs_tickets_df = cs_tickets_df[math.floor(len_of_dataset*config.test_factor):]

    texts = cs_tickets_df['body']

    logger.info(f'Data loaded for testing')
    return cs_tickets_df, texts


def seed_database():
    cs_tickets_df = get_cs_tickets_df()
