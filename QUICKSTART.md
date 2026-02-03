# SmartTicket AI - Quick Start Guide

## Overview
SmartTicket AI is an intelligent customer support system that automatically analyzes and responds to support tickets using:
- **Classification Models**: Categorize tickets by type, priority, and sentiment
- **Vector Search (FAISS)**: Find relevant knowledge base articles
- **RAG (Retrieval-Augmented Generation)**: Generate contextual responses using LLMs

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

### 2. Configure API Keys
Edit `.env` file and add your API key:
```bash
# For OpenAI
OPENAI_API_KEY=sk-your-key-here

# OR for Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-sonnet-20240229
```

### 3. Start the API
```bash
uvicorn api.main:app --reload
```

### 4. Test the API
Visit: http://localhost:8000/docs

## API Examples

### Analyze a Ticket
```bash
curl -X POST "http://localhost:8000/api/v1/ticket/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot login to account",
    "description": "Password reset email not arriving"
  }'
```

Response:
```json
{
  "category": "account",
  "priority": "high",
  "sentiment": "negative",
  "probabilities": {...}
}
```

### Generate Automated Response
```bash
curl -X POST "http://localhost:8000/api/v1/ticket/respond" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "T12345",
    "subject": "Cannot login to account",
    "description": "Password reset email not arriving",
    "customer_email": "user@example.com"
  }'
```

Response:
```json
{
  "ticket_id": "T12345",
  "ticket_analysis": {
    "category": "account",
    "priority": "high",
    "sentiment": "negative"
  },
  "generated_response": "I understand you're having trouble...",
  "confidence_scores": {...}
}
```

### Search Knowledge Base
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to reset password",
    "top_k": 3
  }'
```

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build vector store
python vector_store/build_index.py

# 4. Train models
python training/train.py

# 5. Start API
uvicorn api.main:app --reload
```

## Docker Deployment

```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Project Structure

```
smartticket-ai/
├── data/
│   ├── raw/              # Training data & knowledge base
│   └── processed/        # FAISS index & embeddings
├── training/             # Model training & evaluation
├── inference/            # Prediction & response generation
├── vector_store/         # FAISS vector search
├── api/                  # FastAPI application
├── tests/                # Unit tests
└── docker/               # Docker configuration
```

## Key Files

- `data/raw/knowledge_base.json` - Knowledge base articles
- `data/raw/training_tickets.csv` - Sample training data
- `.env` - Configuration (API keys, model settings)
- `api/main.py` - FastAPI application
- `vector_store/build_index.py` - Build FAISS index
- `training/train.py` - Train classification models

## Adding Custom Knowledge

Edit `data/raw/knowledge_base.json`:
```json
{
  "id": "kb_custom_001",
  "category": "billing",
  "question": "How to view invoices?",
  "answer": "To view your invoices: 1) Login, 2) Go to Billing..."
}
```

Then rebuild the index:
```bash
python vector_store/build_index.py
```

## Customization

### Change LLM Provider
Edit `.env`:
```bash
# Use Anthropic Claude
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-sonnet-20240229

# Use OpenAI
LLM_PROVIDER=openai
MODEL_NAME=gpt-4-turbo-preview
```

### Adjust Response Generation
Edit `.env`:
```bash
TEMPERATURE=0.7        # Creativity (0.0-1.0)
MAX_TOKENS=500        # Response length
TOP_K_RESULTS=3       # Knowledge base articles to use
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_api.py -v
pytest tests/test_models.py -v
```

## Troubleshooting

**Models not found error:**
```bash
python training/train.py
```

**Vector store not found error:**
```bash
python vector_store/build_index.py
```

**API key errors:**
- Check `.env` file exists
- Verify API key is correct
- Ensure LLM_PROVIDER matches your key

## Next Steps

1. **Add more knowledge base articles** in `data/raw/knowledge_base.json`
2. **Train with your own ticket data** by replacing `data/raw/training_tickets.csv`
3. **Customize response templates** in `inference/predict.py`
4. **Deploy to production** using Docker or cloud platform
5. **Integrate with your ticketing system** via the REST API

## Support

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

## License

MIT License
