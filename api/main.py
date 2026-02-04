"""
Main FastAPI Application - SmartTicket AI API
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from typing import List

from .schemas import (
    TicketAnalyzeRequest,
    TicketAnalyzeResponse,
    TicketResponseRequest,
    TicketResponseResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    ErrorResponse,
    KnowledgeBaseItem
)
from .deps import (
    get_settings,
    get_model_loader,
    get_vector_index,
    get_response_generator,
    check_models_loaded,
    check_vector_store_loaded,
    Settings,
    ModelLoader,
    FAISSIndex,
    TicketResponseGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SmartTicket AI",
    description="Intelligent customer support ticket system with automated response generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup"""
    logger.info("Starting SmartTicket AI API...")
    
    # Warm up models
    try:
        logger.info("Loading models...")
        get_model_loader()
        logger.info("Loading vector store...")
        get_vector_index()
        logger.info("Initializing response generator...")
        get_response_generator()
        logger.info("All components loaded successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.warning("API starting with limited functionality")

    yield

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "SmartTicket AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint"""
    models_loaded = check_models_loaded()
    vector_store_loaded = check_vector_store_loaded()
    
    status_code = (
        "healthy" if (models_loaded and vector_store_loaded)
        else "degraded"
    )
    
    return HealthResponse(
        status=status_code,
        version=settings.version,
        models_loaded=models_loaded,
        vector_store_loaded=vector_store_loaded
    )


@app.post(
    "/api/v1/ticket/analyze",
    response_model=TicketAnalyzeResponse,
    tags=["Tickets"],
    status_code=status.HTTP_200_OK
)
async def analyze_ticket(
    request: TicketAnalyzeRequest,
    model_loader: ModelLoader = Depends(get_model_loader)
):
    """
    Analyze a ticket and classify it by category, priority, and sentiment
    
    - **subject**: Ticket subject line
    - **description**: Detailed ticket description
    - **customer_email**: Optional customer email
    """
    try:
        # Combine subject and description
        full_text = f"{request.subject} {request.description}"
        
        # Get predictions
        predictions = model_loader.predict_all(full_text)
        
        # Get probabilities
        probabilities = model_loader.get_prediction_probabilities(full_text)
        
        return TicketAnalyzeResponse(
            category=predictions['category'],
            priority=predictions['priority'],
            sentiment=predictions['sentiment'],
            probabilities=probabilities
        )
        
    except Exception as e:
        logger.error(f"Error analyzing ticket: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing ticket: {str(e)}"
        )


@app.post(
    "/api/v1/ticket/respond",
    response_model=TicketResponseResponse,
    tags=["Tickets"],
    status_code=status.HTTP_200_OK
)
async def generate_ticket_response(
    request: TicketResponseRequest,
    generator: TicketResponseGenerator = Depends(get_response_generator)
):
    """
    Generate an automated response for a ticket using RAG
    
    - **ticket_id**: Optional ticket ID
    - **subject**: Ticket subject line
    - **description**: Detailed ticket description
    - **customer_email**: Optional customer email
    """
    try:
        # Generate response
        result = generator.generate_response(
            subject=request.subject,
            description=request.description,
            customer_email=request.customer_email
        )
        
        # Format response
        return TicketResponseResponse(
            ticket_id=request.ticket_id,
            ticket_analysis=result['ticket_analysis'],
            generated_response=result['generated_response'],
            confidence_scores=result['confidence_scores']
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Knowledge Base"]
)
async def search_knowledge_base(
    request: SearchRequest,
    vector_index: FAISSIndex = Depends(get_vector_index)
):
    """
    Search the knowledge base for relevant information
    
    - **query**: Search query
    - **top_k**: Number of results to return (1-10)
    """
    try:
        # Search vector store
        results = vector_index.search(request.query, top_k=request.top_k)
        
        # Convert to response format
        kb_items = [
            KnowledgeBaseItem(**result)
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=kb_items,
            total_results=len(kb_items)
        )
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching knowledge base: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}")
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, log_level="info")
