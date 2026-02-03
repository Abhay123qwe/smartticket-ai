# SmartTicket AI - Project Structure

```
smartticket-ai/
│
├── 📄 README.md                    # Main documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 pytest.ini                   # Pytest configuration
├── 🔧 setup.sh                     # Automated setup script
│
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── knowledge_base.json     # Knowledge base articles (10 entries)
│   │   └── training_tickets.csv    # Sample training data (20 tickets)
│   └── 📁 processed/
│       └── (FAISS index generated here)
│
├── 📁 training/
│   ├── __init__.py
│   ├── train.py                    # Train classification models
│   └── evaluate.py                 # Evaluate model performance
│
├── 📁 inference/
│   ├── __init__.py
│   ├── model_loader.py             # Load trained models
│   └── predict.py                  # Generate responses with RAG
│
├── 📁 vector_store/
│   ├── __init__.py
│   ├── build_index.py              # Build FAISS vector index
│   └── faiss_index.py              # Query FAISS index
│
├── 📁 api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── schemas.py                  # Pydantic models
│   └── deps.py                     # Dependency injection
│
├── 📁 docker/
│   ├── Dockerfile                  # Docker image configuration
│   └── docker-compose.yml          # Docker Compose setup
│
└── 📁 tests/
    ├── __init__.py
    ├── test_api.py                 # API endpoint tests
    └── test_models.py              # Model component tests
```

## Component Overview

### 1. Data Layer (`data/`)
- **raw/knowledge_base.json**: 10 pre-defined Q&A pairs covering account, billing, technical, features, and security topics
- **raw/training_tickets.csv**: 20 sample tickets for training classification models
- **processed/**: Auto-generated FAISS index and embeddings

### 2. Training Module (`training/`)
- **train.py**: Trains 3 classification models (category, priority, sentiment) using Random Forest
- **evaluate.py**: Generates evaluation reports with accuracy metrics and confusion matrices

### 3. Inference Module (`inference/`)
- **model_loader.py**: Manages loading and caching of trained models
- **predict.py**: Combines classification + RAG to generate automated responses

### 4. Vector Store (`vector_store/`)
- **build_index.py**: Creates FAISS index from knowledge base using sentence transformers
- **faiss_index.py**: Provides search interface for similarity matching

### 5. API Layer (`api/`)
- **main.py**: FastAPI application with 4 main endpoints
- **schemas.py**: Request/response validation models
- **deps.py**: Dependency injection and settings management

### 6. Docker (`docker/`)
- **Dockerfile**: Multi-stage build for production deployment
- **docker-compose.yml**: Complete stack orchestration

### 7. Tests (`tests/`)
- **test_api.py**: Tests all API endpoints
- **test_models.py**: Tests model loading and prediction

## Key Features

✅ **Automated Classification**: Category, priority, and sentiment analysis
✅ **RAG-Powered Responses**: Context-aware responses using vector search
✅ **Dual LLM Support**: Works with OpenAI or Anthropic
✅ **REST API**: Production-ready FastAPI application
✅ **Docker Ready**: Full containerization support
✅ **Comprehensive Tests**: Unit and integration tests included
✅ **Easy Setup**: One-command setup script

## File Count Summary

- Python files: 17
- Configuration files: 6
- Documentation: 2
- Data files: 2
- Docker files: 2
- Test files: 3

**Total: 32 files organized in 9 directories**
