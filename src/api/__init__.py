from fastapi import FastAPI, HTTPException
from db import get_local_session
from models.facebook_bart_large_mnli.load_model import LoadModel
from schemas.pydantic.request_create import RequestCreate
from uuid import uuid4

from schemas.pydantic.postgres_config import PostgresSettings

from db.init_tables import init_tables
from db.requests import add_new_request, update_request

init_tables()

sessionMaker = get_local_session()

model = LoadModel(['technical', 'general', 'billing'])

app = FastAPI()
db = {}

@app.post("/requests")
def create_request(request: RequestCreate):
    # TODO add doc string

    session = sessionMaker()

    # make ui uuid ??
    req_id = add_new_request(request, session)

    # TODO Lazy load the querying to the backend 
    # TODO use the subject too
    responds = model.query(request.content)
    update_request(req_id, responds, session)
    
    return {"id": req_id}

# GET /requests/{id}
# - Returns the stored record with AI fields.
# - Handle not ready yet / Incorrect

# TODO make sure its a uuid ... and also make a return object type fo the results of a doc request
@app.get("/requests/{request_id}")
def create_request(request_id: str):
    if request_id not in db:
        raise HTTPException(status_code=404, detail="Request ID not found")
    return db[request_id]


# Post making dbs

# GET /requests?category=technical
# - Lists filtered records.
# GET /stats (stretch)
# - Returns counts per category over the past seven days.

