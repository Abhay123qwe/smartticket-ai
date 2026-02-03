"""
API Schemas - Pydantic models for request/response validation
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from datetime import datetime


class TicketAnalyzeRequest(BaseModel):
    """Request schema for ticket analysis"""
    subject: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    customer_email: Optional[EmailStr] = None
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "subject": "Cannot login to my account",
                "description": "I forgot my password and the reset email is not arriving",
                "customer_email": "user@example.com"
            }
        }


class TicketAnalyzeResponse(BaseModel):
    """Response schema for ticket analysis"""
    category: str
    priority: str
    sentiment: str
    probabilities: Dict[str, Dict[str, float]]
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "category": "account",
                "priority": "high",
                "sentiment": "negative",
                "probabilities": {
                    "category": {"account": 0.85, "technical": 0.10, "billing": 0.05},
                    "priority": {"high": 0.75, "medium": 0.20, "low": 0.05},
                    "sentiment": {"negative": 0.80, "neutral": 0.15, "positive": 0.05}
                }
            }
        }


class TicketResponseRequest(BaseModel):
    """Request schema for response generation"""
    ticket_id: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    customer_email: Optional[EmailStr] = None
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "ticket_id": "T12345",
                "subject": "Cannot login to my account",
                "description": "I forgot my password and the reset email is not arriving",
                "customer_email": "user@example.com"
            }
        }


class TicketAnalysis(BaseModel):
    """Ticket analysis details"""
    category: str
    priority: str
    sentiment: str


class TicketResponseResponse(BaseModel):
    """Response schema for response generation"""
    ticket_id: Optional[str]
    ticket_analysis: TicketAnalysis
    generated_response: str
    confidence_scores: Dict[str, Dict[str, float]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "ticket_id": "T12345",
                "ticket_analysis": {
                    "category": "account",
                    "priority": "high",
                    "sentiment": "negative"
                },
                "generated_response": "I understand you're having trouble accessing your account...",
                "confidence_scores": {
                    "category": {"account": 0.85},
                    "priority": {"high": 0.75},
                    "sentiment": {"negative": 0.80}
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class KnowledgeBaseItem(BaseModel):
    """Knowledge base entry"""
    id: str
    category: str
    question: str
    answer: str
    score: Optional[float] = None
    rank: Optional[int] = None


class SearchRequest(BaseModel):
    """Request schema for knowledge base search"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "query": "How to reset password",
                "top_k": 3
            }
        }


class SearchResponse(BaseModel):
    """Response schema for knowledge base search"""
    query: str
    results: List[KnowledgeBaseItem]
    total_results: int
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "query": "How to reset password",
                "results": [
                    {
                        "id": "kb_001",
                        "category": "account",
                        "question": "How to reset password?",
                        "answer": "To reset your password...",
                        "score": 0.95,
                        "rank": 1
                    }
                ],
                "total_results": 1
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models_loaded: bool
    vector_store_loaded: bool
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00",
                "models_loaded": True,
                "vector_store_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred",
                "timestamp": "2024-01-01T12:00:00"
            }
        }
