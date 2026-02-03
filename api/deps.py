"""
Dependencies - Dependency injection for FastAPI
"""
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path
from inference.model_loader import ModelLoader
from inference.predict import TicketResponseGenerator
from vector_store.faiss_index import FAISSIndex

load_dotenv()


class Settings:
    """Application settings"""
    
    def __init__(self):
        self.app_name: str = "SmartTicket AI"
        self.version: str = "1.0.0"
        self.api_version: str = os.getenv("API_VERSION", "v1")
        
        # LLM settings
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
        self.model_name: str = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))
        
        # Paths
        self.models_dir: str = os.getenv("CLASSIFIER_MODEL_PATH", "models")
        self.vector_store_path: str = os.getenv(
            "VECTOR_STORE_PATH",
            "data/processed/faiss_index"
        )
        
        # API settings
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_model_loader() -> ModelLoader:
    """Get cached model loader instance"""
    settings = get_settings()
    return ModelLoader(models_dir=Path("models"))


@lru_cache()
def get_vector_index() -> FAISSIndex:
    """Get cached FAISS index instance"""
    settings = get_settings()
    return FAISSIndex(index_path=settings.vector_store_path)


@lru_cache()
def get_response_generator() -> TicketResponseGenerator:
    """Get cached response generator instance"""
    settings = get_settings()
    return TicketResponseGenerator(
        llm_provider=settings.llm_provider,
        model_name=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )


def check_models_loaded() -> bool:
    """Check if models are loaded successfully"""
    try:
        loader = get_model_loader()
        loader.load_classifier_models()
        loader.load_encoders()
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def check_vector_store_loaded() -> bool:
    """Check if vector store is loaded successfully"""
    try:
        index = get_vector_index()
        return index.index is not None
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return False
