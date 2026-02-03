"""Vector store module for FAISS-based similarity search"""

from .faiss_index import FAISSIndex
from .build_index import VectorStoreBuilder

__all__ = ['FAISSIndex', 'VectorStoreBuilder']
