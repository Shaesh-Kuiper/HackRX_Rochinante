# rag_server.py - FastAPI server wrapper for RAG pipeline
import asyncio
import os
import sys
import time
from typing import List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the modified rag_app module
from rag_app import process_rag_pipeline

# Configuration
EXPECTED_AUTH_TOKEN = "0d40085aa1ab7502b99a71688edfb832121051dc7afd7ad9c3f4acff3c4fe176"
RERANKER_URL = os.getenv("RERANKER_URL", "https://bb5e5fbf2d8c.ngrok-free.app/rerank")
security = HTTPBearer(auto_error=False)  # This adds the scheme to Swagger

# Request/Response Models
class RAGRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class RAGResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    reranker_url: str

# Create FastAPI app
app = FastAPI(
    title="High-Performance RAG API",
    description="Sub-20s RAG pipeline with caching and multi-format support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming {request.method} request to {request.url.path}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Request completed in {process_time:.3f}s with status {response.status_code}")
    
    return response

@app.get("/health", response_model=HealthResponse)
@app.head("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        reranker_url=RERANKER_URL
    )

async def process_rag_request(request: RAGRequest) -> RAGResponse:
    """Process RAG request and return answers"""
    start_time = time.time()
    
    # Verbose logging
    logger.info("="*80)
    logger.info("📥 NEW RAG REQUEST RECEIVED")
    logger.info("="*80)
    logger.info(f"📄 Document URL: {request.documents[:100]}{'...' if len(request.documents) > 100 else ''}")
    logger.info(f"❓ Number of questions: {len(request.questions)}")
    
    # Log first few questions for debugging
    for i, q in enumerate(request.questions, start=1):
        logger.info(f"   Q{i}: {q}")
    # if len(request.questions) > 3:
    #     logger.info(f"   ... and {len(request.questions) - 3} more questions")
    logger.info("="*80)
    
    try:
        # Process through RAG pipeline
        results = await process_rag_pipeline(
            document_url=request.documents,
            questions=request.questions,
            reranker_url=RERANKER_URL
        )
        
        # Extract answers
        answers = [result["answer"] for result in results]
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"✅ RAG processing completed in {total_time:.2f}s")
        logger.info(f"📊 Processed {len(answers)} answers")
        
        return RAGResponse(answers=answers)
        
    except ValueError as e:
        # Handle unsupported file types or other value errors
        error_msg = str(e)
        logger.error(f"❌ ValueError: {error_msg}")
        
        if "not supported" in error_msg or "too large" in error_msg:
            raise HTTPException(status_code=400, detail=error_msg)
        raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/hackrx/run", response_model=RAGResponse)
async def hackrx_run(
    request: RAGRequest,
    creds: HTTPAuthorizationCredentials = Depends(security)
):
    """Main RAG endpoint"""
    # Verify authorization
    token = creds.credentials if creds else None
    if token != EXPECTED_AUTH_TOKEN:
        logger.warning("❌ Unauthorized request attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return await process_rag_request(request)

@app.post("/api/v1/hackrx/run", response_model=RAGResponse)
async def hackrx_run_v1(
    request: RAGRequest,
    creds: HTTPAuthorizationCredentials = Depends(security)
):
    """API v1 RAG endpoint (same as /hackrx/run)"""
    # Verify authorization
    token = creds.credentials if creds else None
    if token != EXPECTED_AUTH_TOKEN:
        logger.warning("❌ Unauthorized request attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return await process_rag_request(request)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "High-Performance RAG API",
        "version": "1.0.0",
        "endpoints": {
            "rag": ["/hackrx/run", "/api/v1/hackrx/run"],
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # Set event loop policy for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run server
    uvicorn.run(
        "rag_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )