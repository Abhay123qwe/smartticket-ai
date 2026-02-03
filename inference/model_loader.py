"""
Model Loader - Load trained models for inference
"""
import os
import pickle
from typing import Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class ModelLoader:
    """Manages loading and caching of trained models"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (BASE_DIR / "models")

        self._models_cache = {}
        self._encoders_cache = None
        
    def load_classifier_models(self) -> Dict:
        """Load all classification models"""
        if 'classifiers' in self._models_cache:
            return self._models_cache['classifiers']
        
        models_path = Path(self.models_dir)
        
        if not models_path.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}. "
                "Please run training/train.py first."
            )
        
        models = {}
        
        # Load category model
        with open(models_path / "category_model.pkl", 'rb') as f:
            models['category'] = pickle.load(f)
        
        # Load priority model
        with open(models_path / "priority_model.pkl", 'rb') as f:
            models['priority'] = pickle.load(f)
        
        # Load sentiment model
        with open(models_path / "sentiment_model.pkl", 'rb') as f:
            models['sentiment'] = pickle.load(f)
        
        self._models_cache['classifiers'] = models
        print("Classification models loaded successfully")
        
        return models
    
    def load_encoders(self) -> Dict:
        """Load label encoders"""
        if self._encoders_cache is not None:
            return self._encoders_cache
        
        encoders_path = Path(self.models_dir) / "encoders.pkl"
        
        if not encoders_path.exists():
            raise FileNotFoundError(
                f"Encoders not found: {encoders_path}. "
                "Please run training/train.py first."
            )
        
        with open(encoders_path, 'rb') as f:
            self._encoders_cache = pickle.load(f)
        
        print("Label encoders loaded successfully")
        
        return self._encoders_cache
    
    def predict_category(self, text: str) -> str:
        """Predict ticket category"""
        models = self.load_classifier_models()
        encoders = self.load_encoders()
        
        prediction_idx = models['category'].predict([text])[0]
        category = encoders['category'].inverse_transform([prediction_idx])[0]
        
        return category
    
    def predict_priority(self, text: str) -> str:
        """Predict ticket priority"""
        models = self.load_classifier_models()
        encoders = self.load_encoders()
        
        prediction_idx = models['priority'].predict([text])[0]
        priority = encoders['priority'].inverse_transform([prediction_idx])[0]
        
        return priority
    
    def predict_sentiment(self, text: str) -> str:
        """Predict ticket sentiment"""
        models = self.load_classifier_models()
        encoders = self.load_encoders()
        
        prediction_idx = models['sentiment'].predict([text])[0]
        sentiment = encoders['sentiment'].inverse_transform([prediction_idx])[0]
        
        return sentiment
    
    def predict_all(self, text: str) -> Dict[str, str]:
        """Predict category, priority, and sentiment"""
        models = self.load_classifier_models()
        encoders = self.load_encoders()
        
        # Get predictions
        category_idx = models['category'].predict([text])[0]
        priority_idx = models['priority'].predict([text])[0]
        sentiment_idx = models['sentiment'].predict([text])[0]
        
        # Decode predictions
        return {
            'category': encoders['category'].inverse_transform([category_idx])[0],
            'priority': encoders['priority'].inverse_transform([priority_idx])[0],
            'sentiment': encoders['sentiment'].inverse_transform([sentiment_idx])[0]
        }
    
    def get_prediction_probabilities(self, text: str) -> Dict:
        """Get prediction probabilities for all classes"""
        models = self.load_classifier_models()
        encoders = self.load_encoders()
        
        # Get probability predictions
        cat_probs = models['category'].predict_proba([text])[0]
        pri_probs = models['priority'].predict_proba([text])[0]
        sent_probs = models['sentiment'].predict_proba([text])[0]
        
        return {
            'category': {
                cls: float(prob)
                for cls, prob in zip(encoders['category'].classes_, cat_probs)
            },
            'priority': {
                cls: float(prob)
                for cls, prob in zip(encoders['priority'].classes_, pri_probs)
            },
            'sentiment': {
                cls: float(prob)
                for cls, prob in zip(encoders['sentiment'].classes_, sent_probs)
            }
        }


if __name__ == "__main__":
    # Test model loading
    loader = ModelLoader()
    
    test_text = "My account is locked and I can't access it"
    
    print(f"Test Text: {test_text}\n")
    
    predictions = loader.predict_all(test_text)
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value}")
    
    print("\nProbabilities:")
    probs = loader.get_prediction_probabilities(test_text)
    for model_name, class_probs in probs.items():
        print(f"\n{model_name}:")
        for cls, prob in class_probs.items():
            print(f"  {cls}: {prob:.4f}")
