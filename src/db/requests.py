from datetime import datetime
from sqlalchemy import update
from schemas.db.support_request import SupportRequest
from schemas.pydantic.model_response import AiResponse
from schemas.pydantic.request import RequestCreate

def add_new_request(request: RequestCreate, session):
    # Create a new request
    if request.subject:
        new_request = SupportRequest(
            subject=request.subject,
            body=request.content,
        )
    else:
        new_request = SupportRequest(
            body=request.content,
        )


    # Add and commit
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