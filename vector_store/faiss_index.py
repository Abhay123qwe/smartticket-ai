"""
FAISS Index - Query interface for vector similarity search
"""
import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class FAISSIndex:
    """FAISS vector store for similarity search"""
    
    def __init__(
        self,
        index_path: str | None = None,
        embedding_model_name: str | None = None,
        top_k: int | None = None
    ):
        self.index_path = index_path or os.getenv(
            "VECTOR_STORE_PATH",
            "data/processed/faiss_index"
        )
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k = top_k or int(os.getenv("TOP_K_RESULTS", "3"))
        
        # Load model and index
        self.model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.metadata = None
        self._load_index()
        
    def _load_index(self):
        """Load FAISS index and metadata"""
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Please run vector_store/build_index.py first."
            )
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
    def search(
        self,
        query: str,
        top_k: int | None = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with knowledge base entries and scores
        """
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search index
        if self.index is None:
            raise ValueError("Index is not initialized")

        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        if self.metadata is None:
            raise ValueError("Metadata is not initialized")
        for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index < len(self.metadata):
                result = self.metadata[index].copy()
                result['score'] = float(1 / (1 + distance))  # Convert distance to similarity
                result['rank'] = idx + 1
                results.append(result)
        
        return results
    
    def get_context(
        self,
        query: str,
        top_k: int | None = None
    ) -> str:
        """
        Get formatted context string for RAG
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        context_parts = []
        for result in results:
            context_parts.append(
                f"Category: {result['category']}\n"
                f"Question: {result['question']}\n"
                f"Answer: {result['answer']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def add_to_index(self, text: str, metadata: Dict):
        """
        Add new entry to index (for dynamic updates)
        
        Args:
            text: Text to embed and add
            metadata: Metadata for the entry
        """
        # Encode text
        embedding = self.model.encode([text], convert_to_numpy=True)
        
        # Add to index
        if self.index is not None:
            self.index.add(embedding.astype('float32'))
        
        # Add metadata
        if self.metadata is not None:
            self.metadata.append(metadata)

        if self.index is not None:
            print(f"Added new entry to index. Total vectors: {self.index.ntotal}")


if __name__ == "__main__":
    # Test the index
    faiss_index = FAISSIndex()
    
    test_query = "I can't login to my account"
    print(f"\nTest Query: {test_query}\n")
    
    results = faiss_index.search(test_query, top_k=3)
    
    print("Top Results:")
    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Category: {result['category']}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer'][:100]}...")
