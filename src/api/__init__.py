from fastapi import BackgroundTasks, FastAPI, HTTPException
from db import get_local_session
from models.facebook_bart_large_mnli import MnliModel
from schemas.pydantic.request import RequestCreate, RequestResponse
from uuid import uuid4

from schemas.pydantic.postgres_config import PostgresSettings

from db.init_tables import init_tables
from db.requests import add_new_request, update_request, get_request

sessionMaker = get_local_session()
model = MnliModel(['technical', 'general', 'billing'])
app = FastAPI()

def process_ai_request(req_id, request):
    session = sessionMaker()
    # TODO use the subject too
    responds = model.query(request.content)
    update_request(req_id, responds, session)

@app.post("/requests")
def create_request(request: RequestCreate, background_tasks: BackgroundTasks) -> RequestResponse:
    # TODO add doc string

    session = sessionMaker()

    # make ui uuid ??
    req_id = add_new_request(request, session)

    background_tasks.add_task(process_ai_request, req_id, request)
    
    return RequestResponse(id=req_id)

# GET /requests/{id}
# TODO make sure its a uuid ... check id ... and also make a return object type fo the results of a doc request
@app.get("/requests/{request_id}")
def create_request(request_id: int) -> RequestResponse:
    session = sessionMaker()
    result = get_request(request_id, session)
    if not result:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    return RequestResponse(
        id=result.id,
        resolved=result.resolved,
        resolved_at=result.resolved_at,
        subject=result.subject,
        summary=result.summary,
        body=result.body,
        predicted_category=result.predicted_category,
        confidence=result.confidence
    )

# GET /requests?category=technical
# - Lists filtered records.

# GET /stats (stretch)
# - Returns counts per category over the past seven days.

