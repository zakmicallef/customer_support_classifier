from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from db import get_engine, get_session_maker
from models import get_model
from schemas.pydantic.request import CategoryQueryParams, RequestCreate, RequestResponse

from db.requests import add_new_request, get_requests_last_week, update_request, get_request
from util.load_configs import load_live_config

sessionMaker = get_session_maker()
config = load_live_config()
priority_model, queue_model = get_model(config)
app = FastAPI()

def process_ai_request(req_id, request: RequestCreate):
    session = sessionMaker()
    responds = queue_model.query(request.get_info)
    update_request(req_id, responds, session)

@app.post(
    "/requests",
    response_model=RequestResponse,
    summary="Predict Request Queue Category and Priority",
    description="Create a new request to be processed by the AI model. Returns the request ID. Predicts category and priority.",
    tags=["requests", "predict category", "predict priority"],
)
def create_request(request: RequestCreate, background_tasks: BackgroundTasks) -> RequestResponse:
    session = sessionMaker()
    req_id = add_new_request(request, session)
    background_tasks.add_task(process_ai_request, req_id, request)
    return RequestResponse(id=req_id)

@app.get(
    "/requests",
    response_model=list[RequestResponse],
    summary="Get Requests",
    description="Retrieve all requests from the past week. Returns a list of requests with their details. filtering by category.",
    tags=["requests", "historical"],
)
def get_requests(params: CategoryQueryParams = Depends()) -> list[RequestResponse]:
    engine = get_engine()
    df = get_requests_last_week(engine, params.category)
    return [RequestResponse(**row) for row in df.to_dict(orient="records")]

@app.get(
    "/requests/{request_id}",
    response_model=RequestResponse,
    summary="Get Request by ID",
    description="Retrieve a specific request by its ID. Returns the request details including category and priority. Also returns the resolved status and timestamps.",
    tags=["requests", "historical"],
)
def get_request_by_id(request_id: int) -> RequestResponse:
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

# GET /stats (stretch)
# - Returns counts per category over the past seven days.

