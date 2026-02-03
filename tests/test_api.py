"""
Test API Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "SmartTicket AI API" in response.json()["message"]


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_analyze_ticket():
    """Test ticket analysis endpoint"""
    payload = {
        "subject": "Cannot login to account",
        "description": "I forgot my password and reset email not arriving",
        "customer_email": "test@example.com"
    }
    
    response = client.post("/api/v1/ticket/analyze", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "category" in data
    assert "priority" in data
    assert "sentiment" in data
    assert "probabilities" in data


def test_analyze_ticket_missing_field():
    """Test ticket analysis with missing required field"""
    payload = {
        "subject": "Test subject"
        # Missing description
    }
    
    response = client.post("/api/v1/ticket/analyze", json=payload)
    assert response.status_code == 422  # Validation error


def test_search_knowledge_base():
    """Test knowledge base search endpoint"""
    payload = {
        "query": "How to reset password",
        "top_k": 3
    }
    
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total_results" in data
    assert len(data["results"]) <= 3


def test_search_invalid_top_k():
    """Test search with invalid top_k value"""
    payload = {
        "query": "test query",
        "top_k": 20  # Exceeds max of 10
    }
    
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.skip(reason="Requires LLM API keys")
def test_generate_response():
    """Test response generation endpoint (requires API keys)"""
    payload = {
        "ticket_id": "T12345",
        "subject": "Cannot login to account",
        "description": "I forgot my password and reset email not arriving",
        "customer_email": "test@example.com"
    }
    
    response = client.post("/api/v1/ticket/respond", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "ticket_analysis" in data
    assert "generated_response" in data
    assert "confidence_scores" in data
    assert data["ticket_id"] == "T12345"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
