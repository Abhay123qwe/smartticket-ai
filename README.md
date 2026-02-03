# SmartTicket AI

An intelligent customer support ticket system that combines RAG (Retrieval-Augmented Generation) with automated response generation.

## Features

- 🎯 Automated ticket classification
- 🤖 AI-powered response generation using RAG
- 🔍 Vector similarity search with FAISS
- 📊 Ticket priority and sentiment analysis
- 🚀 FastAPI REST API
- 🐳 Docker support

## Architecture

The system combines:
- **Vector Store (FAISS)**: Stores embedded knowledge base for similarity search
- **LLM (OpenAI/Anthropic)**: Generates contextual responses based on retrieved context
- **Classification Model**: Categorizes tickets by type and priority

## Setup

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Build the vector store:
```bash
python vector_store/build_index.py
```

## Usage

### Training
```bash
python training/train.py
python training/evaluate.py
```

### Running the API
```bash
uvicorn api.main:app --reload
```

### Docker
```bash
docker build -f docker/Dockerfile -t smartticket-ai .
docker run -p 8000:8000 --env-file .env smartticket-ai
```

## API Endpoints

- `POST /api/v1/ticket/analyze` - Analyze a new ticket
- `POST /api/v1/ticket/respond` - Generate automated response
- `GET /api/v1/health` - Health check

