"""
Vector Store Builder - Creates and manages FAISS index for knowledge base
"""
import json
from operator import index
import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from openai import embeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class VectorStoreBuilder:
    """Builds and saves FAISS vector index from knowledge base"""
    
    def __init__(
        self,
        embedding_model_name: str | None = None,
        knowledge_base_path: str | None = None,
        output_path: str | None = None
    ):
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.knowledge_base_path = knowledge_base_path or os.getenv(
            "KNOWLEDGE_BASE_PATH",
            "data/raw/knowledge_base.json"
        )
        self.output_path = output_path or os.getenv(
            "VECTOR_STORE_PATH",
            "data/processed/faiss_index"
        )
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base from JSON file"""
        print(f"Loading knowledge base from: {self.knowledge_base_path}")
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        print(f"Loaded {len(knowledge_base)} knowledge base entries")
        return knowledge_base
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text list"""
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings"""
        print(f"Building FAISS index with dimension {self.dimension}")

        faiss_embeddings = np.asarray(embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(self.dimension)
        index.add(faiss_embeddings) # type: ignore

        print(f"Index built with {index.ntotal} vectors")
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        knowledge_base: List[Dict],
        embeddings: np.ndarray
    ):
        """Save FAISS index and metadata"""
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(self.output_path, "index.faiss")
        faiss.write_index(index, index_file)
        print(f"Saved FAISS index to: {index_file}")
        
        # Save knowledge base metadata
        metadata_file = os.path.join(self.output_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(knowledge_base, f)
        print(f"Saved metadata to: {metadata_file}")
        
        # Save embeddings for reference
        embeddings_file = os.path.join(self.output_path, "embeddings.npy")
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to: {embeddings_file}")
        
    def build(self):
        """Main build process"""
        print("\n=== Starting Vector Store Build ===\n")
        
        # Load knowledge base
        knowledge_base = self.load_knowledge_base()
        
        # Prepare texts for embedding (combine question + answer)
        texts = [
            f"{item['question']} {item['answer']}"
            for item in knowledge_base
        ]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Build FAISS index
        index = self.build_index(embeddings)
        
        # Save everything
        self.save_index(index, knowledge_base, embeddings)
        
        print("\n=== Vector Store Build Complete ===\n")
        print(f"Index location: {self.output_path}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Dimension: {self.dimension}")


if __name__ == "__main__":
    builder = VectorStoreBuilder()
    builder.build()
