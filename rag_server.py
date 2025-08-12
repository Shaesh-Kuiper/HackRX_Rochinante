import asyncio
import os
import time
import logging
from typing import List, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rag_app import process_rag_pipeline

EXPECTED_AUTH_TOKEN = "0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176"
security = HTTPBearer(auto_error=False)

class RAGRequest(BaseModel):
    documents: Union[str, List[str]] = Field(..., description="URL or list of URLs of the documents to process")
    questions: Union[str, List[str]] = Field(..., description="Question or list of questions to answer")

class RAGResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str

app = FastAPI(title="High-Performance RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} completed in {time.perf_counter() - start_time:.3f}s")
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())

async def process_rag_request(request: RAGRequest) -> RAGResponse:
    start_time = time.perf_counter()
    
    doc_urls = [request.documents] if isinstance(request.documents, str) else request.documents
    qs = [request.questions] if isinstance(request.questions, str) else request.questions
    
    logger.info(f"Processing {len(doc_urls)} document(s) for {len(qs)} question(s).")
    
    try:
        results = await process_rag_pipeline(
            document_urls=doc_urls,
            questions=qs
        )
        answers = [res["answer"] for res in results]
        logger.info(f"✅Processing completed in {time.perf_counter() - start_time:.2f}s")
        return RAGResponse(answers=answers)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/hackrx/run", response_model=RAGResponse)
@app.post("/api/v1/hackrx/run", response_model=RAGResponse)
async def hackrx_run(request: RAGRequest, creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials if creds else None
    if token != EXPECTED_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await process_rag_request(request)

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=True)