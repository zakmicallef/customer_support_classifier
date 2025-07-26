from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import select, update
from schemas.db.support_request import SupportRequest
from schemas.pydantic.model_response import AiResponse
from schemas.pydantic.request import RequestCreate

def add_new_request(request: RequestCreate, session):
    if request.subject:
        new_request = SupportRequest(
            subject=request.subject,
            body=request.content,
        )
    else:
        new_request = SupportRequest(
            body=request.content,
        )

    session.add(new_request)
    session.commit()
    session.refresh(new_request)

    return new_request.id

def update_request(id_, response: AiResponse, session):
    stmt = (
        update(SupportRequest)
        .where(SupportRequest.id == id_)
        .values(
            id = id_,
            resolved = True,
            predicted_category = response.category,
            confidence = response.confidence,
            summary = response.summary,
            resolved_at = datetime.now()
        )
    )
    session.execute(stmt)
    session.commit()

def get_request(id, session):
    return session.get(SupportRequest, id)


def get_requests_last_week(engine, category: str = None):
    week_ago = datetime.now() - timedelta(days=7)
    base_query = (
        select(
            SupportRequest.id,
            SupportRequest.subject,
            SupportRequest.body,
            SupportRequest.resolved,
            SupportRequest.predicted_category,
            SupportRequest.confidence,
            SupportRequest.summary,
            SupportRequest.created_at,
            SupportRequest.resolved_at,
            SupportRequest.status
        )
        .where(SupportRequest.created_at >= week_ago)
    )

    if category:
        base_query = base_query.where(SupportRequest.predicted_category == category)

    query = base_query.order_by(SupportRequest.created_at.desc())
    df = pd.read_sql_query(query, con=engine)
    return df