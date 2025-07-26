# TODO Move to a better place
from loguru import logger
import pandas as pd
from schemas.db.customer_support_dataset import Language, Priority, Queue, Tag, Ticket
from util.load_configs import load_data_config
import math

import pandas as pd
import math
from sqlalchemy import select, func
from schemas.db.customer_support_dataset import Ticket, Priority, Queue, Language, Tag, ticket_tags

def get_dataset(engine, session, set='train', tags=False, language='en'):
    logger.info(f"Getting dataset for set: {set}, language: {language}, tags: {tags}")
    config = load_data_config()

    if tags:
        tag_subquery = (
            select(
                ticket_tags.c.ticket_id.label("ticket_id"),
                func.array_agg(Tag.name).label("tags")
            )
            .join(Tag, Tag.id == ticket_tags.c.tag_id)
            .group_by(ticket_tags.c.ticket_id)
            .subquery()
        )

        if language != 'all':
            base_query = (
                select(
                    Ticket.id,
                    Ticket.subject,
                    Ticket.body,
                    Ticket.answer,
                    Ticket.type,
                    Priority.name.label("priority"),
                    Queue.name.label("queue"),
                    Language.name.label("language"),
                    tag_subquery.c.tags
                )
                .outerjoin(Priority, Ticket.priority_id == Priority.id)
                .outerjoin(Queue, Ticket.queue_id == Queue.id)
                .outerjoin(Language, Ticket.language_id == Language.id)
                .outerjoin(tag_subquery, tag_subquery.c.ticket_id == Ticket.id)
                .where(Language.name == language)
            )
        else:
            base_query = (
                select(
                    Ticket.id,
                    Ticket.subject,
                    Ticket.body,
                    Ticket.answer,
                    Ticket.type,
                    Priority.name.label("priority"),
                    Queue.name.label("queue"),
                    Language.name.label("language"),
                    tag_subquery.c.tags
                )
                .outerjoin(Priority, Ticket.priority_id == Priority.id)
                .outerjoin(Queue, Ticket.queue_id == Queue.id)
                .outerjoin(Language, Ticket.language_id == Language.id)
                .outerjoin(tag_subquery, tag_subquery.c.ticket_id == Ticket.id)
            )
    else:
        if language != 'all':
            base_query = (
                select(
                    Ticket.id,
                    Ticket.subject,
                    Ticket.body,
                    Ticket.answer,
                    Ticket.type,
                    Priority.name.label("priority"),
                    Queue.name.label("queue"),
                    Language.name.label("language")
                )
                .outerjoin(Priority, Ticket.priority_id == Priority.id)
                .outerjoin(Queue, Ticket.queue_id == Queue.id)
                .outerjoin(Language, Ticket.language_id == Language.id)
                .where(Language.name == language)
            )
        else:
            base_query = (
                select(
                    Ticket.id,
                    Ticket.subject,
                    Ticket.body,
                    Ticket.answer,
                    Ticket.type,
                    Priority.name.label("priority"),
                    Queue.name.label("queue"),
                    Language.name.label("language")
                )
                .outerjoin(Priority, Ticket.priority_id == Priority.id)
                .outerjoin(Queue, Ticket.queue_id == Queue.id)
                .outerjoin(Language, Ticket.language_id == Language.id)
            )

    if set != 'all':
        if language == 'all':
            total = session.query(func.count(Ticket.id)).scalar()
        else:
            total = session.query(func.count(Ticket.id))\
                .join(Language, Ticket.language_id == Language.id)\
                .filter(Language.name == language)\
                .scalar()

        split_index = math.floor(total * config.test_factor)
        
        if set == 'train':
            query = base_query.limit(total - split_index).offset(0)
        else:
            query = base_query.limit(split_index).offset(total - split_index)

        logger.info(f"Total records: {total}, Split index: {split_index}, offset: {total - split_index}")

    return pd.read_sql_query(query, con=engine)

def get_or_create(model, name, cache, session):
    if pd.isna(name):
        return None
    if name not in cache:
        obj = session.query(model).filter_by(name=name).first()
        if not obj:
            obj = model(name=name)
            session.add(obj)
            session.flush()
        cache[name] = obj
    return cache[name]

def seed_database_from_df(df, session):
    priority_map = {}
    queue_map = {}
    language_map = {}
    tag_map = {}

    for _, row in df.iterrows():
        priority = get_or_create(Priority, row['priority'], priority_map, session)
        queue = get_or_create(Queue, row['queue'], queue_map, session)
        language = get_or_create(Language, row['language'], language_map, session)

        tags = []
        for i in range(1, 9):
            tag_col = f'tag_{i}'
            tag_value = row.get(tag_col)
            if pd.notna(tag_value):
                tag = get_or_create(Tag, tag_value, tag_map, session)
                tags.append(tag)

        ticket = Ticket(
            subject=row['subject'],
            body=row['body'],
            answer=row['answer'],
            type=row['type'],
            priority=priority,
            queue=queue,
            language=language,
            tags=tags
        )
        session.add(ticket)
    session.commit()
