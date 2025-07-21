from fastapi import FastAPI, HTTPException
from models.facebook_bart_large_mnli.load_model import LoadModel
from schemas.request_create import RequestCreate
from uuid import uuid4

model = LoadModel(['technical', 'general', 'billing'])

app = FastAPI()
db = {}

@app.post("/requests")
def create_request(request: RequestCreate):
    # TODO add doc string

    req_id = str(uuid4())
    # TODO Lazy load the querying to the backend 
    responds = model.query(request.text)
    db[req_id] = {
        "id": req_id,
        "text": request.text or f"{request.subject or ''} {request.body or ''}".strip(),
        "category": responds.category,  # to be filled by AI later
        "confidence": responds.confidence, # TODO round this number
        "summary": responds.summary,
    }
    print(db[req_id])
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

