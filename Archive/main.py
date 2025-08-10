# main.py
from fastapi import FastAPI, Header, HTTPException, Depends, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import List, Union, Any

# ───────── CONFIG ──────────
TEAM_TOKEN = "0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176"

security = HTTPBearer(auto_error=False)   # adds the scheme to Swagger

# ───────── DATA MODELS ─────
class RunRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]
    
    @validator('documents', pre=True)
    def validate_documents(cls, v: Any) -> Union[str, List[str]]:
        # Handle string input
        if isinstance(v, str):
            # If it's a JSON-like list string, try to parse it first
            if v.strip().startswith('[') and v.strip().endswith(']'):
                try:
                    import json
                    parsed = json.loads(v)
                    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                        return parsed
                except json.JSONDecodeError:
                    pass
            # Otherwise treat it as a single document string
            return v

        # Handle list input
        if isinstance(v, list):
            if all(isinstance(item, str) for item in v):
                return v
            raise ValueError("All items in documents list must be strings")

        raise ValueError("documents must be a string or list of strings")

class RunResponse(BaseModel):
    answers: List[str]

# ───────── APP ─────────────
app = FastAPI(title="HackRX RAG Service")

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_head():
    return Response(status_code=200)

# import your pipeline objects here
from rag_pipeline import answer_questions   # see §3
@app.post("/api/v1/hackrx/run",  response_model=RunResponse)
@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(
        payload: RunRequest,
        creds: HTTPAuthorizationCredentials = Depends(security)
    ):
    token = creds.credentials if creds else None
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Add debugging (remove after fixing)
    print(f"Received documents type: {type(payload.documents)}")
    print(f"Documents value: {payload.documents}")
    
    # 2.  Normalise input
    docs = payload.documents
    doc_urls = docs if isinstance(docs, list) else [docs]

    # 3.  Run the pipeline
    answers = await answer_questions(doc_urls, payload.questions)

    return RunResponse(answers=answers)

@app.on_event("shutdown")
async def _shutdown_cleanup():
    await net_utils.shared_session.close()
