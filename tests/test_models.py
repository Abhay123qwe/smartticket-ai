"""
Test Model Components
"""
import pytest
import os
from pathlib import Path


class TestVectorStore:
    """Test vector store functionality"""
    
    @pytest.mark.skipif(
        not Path("data/processed/faiss_index/index.faiss").exists(),
        reason="Vector store not built"
    )
    def test_faiss_index_load(self):
        """Test loading FAISS index"""
        from vector_store.faiss_index import FAISSIndex
        
        index = FAISSIndex()
        assert index.index is not None
        assert index.metadata is not None
    
    @pytest.mark.skipif(
        not Path("data/processed/faiss_index/index.faiss").exists(),
        reason="Vector store not built"
    )
    def test_search_functionality(self):
        """Test vector search"""
        from vector_store.faiss_index import FAISSIndex
        
        index = FAISSIndex()
        results = index.search("password reset", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        assert all('score' in r for r in results)
        assert all('rank' in r for r in results)


class TestModelLoader:
    """Test model loading functionality"""
    
    @pytest.mark.skipif(
        not Path("models/category_model.pkl").exists(),
        reason="Models not trained"
    )
    def test_load_models(self):
        """Test loading trained models"""
        from inference.model_loader import ModelLoader
        
        loader = ModelLoader()
        models = loader.load_classifier_models()
        
        assert 'category' in models
        assert 'priority' in models
        assert 'sentiment' in models
    
    @pytest.mark.skipif(
        not Path("models/category_model.pkl").exists(),
        reason="Models not trained"
    )
    def test_predict_all(self):
        """Test prediction functionality"""
        from inference.model_loader import ModelLoader
        
        loader = ModelLoader()
        test_text = "I cannot access my account"
        
        predictions = loader.predict_all(test_text)
        
        assert 'category' in predictions
        assert 'priority' in predictions
        assert 'sentiment' in predictions
        assert isinstance(predictions['category'], str)
        assert isinstance(predictions['priority'], str)
        assert isinstance(predictions['sentiment'], str)


class TestTraining:
    """Test training functionality"""
    
    def test_data_loading(self):
        """Test loading training data"""
        from training.train import TicketClassifier
        
        classifier = TicketClassifier()
        df = classifier.load_data()
        
        assert len(df) > 0
        assert 'subject' in df.columns
        assert 'description' in df.columns
        assert 'category' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
