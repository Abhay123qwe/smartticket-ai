"""
Training Script - Train ticket classification model
"""
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()


class TicketClassifier:
    """Multi-output classifier for ticket categorization"""
    
    def __init__(self, data_path: str|None = None, output_dir: str|None = None):
        self.data_path = data_path or "data/raw/training_tickets.csv"
        self.output_dir = output_dir or "models"
        
        # Label encoders
        self.category_encoder = LabelEncoder()
        self.priority_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()
        
        # Models
        
        self.category_model = None
        self.priority_model = None
        self.sentiment_model = None
        
            
        
    def load_data(self) -> pd.DataFrame:
        """Load training data"""
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} training examples")
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare text features and labels"""
        # Combine subject and description for features
        X = df["subject"].str.cat(df["description"], sep=" ")

        
        # Encode labels
        y_category = self.category_encoder.fit_transform(df['category'])
        y_priority = self.priority_encoder.fit_transform(df['priority'])
        y_sentiment = self.sentiment_encoder.fit_transform(df['sentiment'])
        
        return X, y_category, y_priority, y_sentiment
    
    def build_pipeline(self) -> Pipeline:
        """Build classification pipeline"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def train(self):
        """Train all classification models"""
        print("\n=== Starting Model Training ===\n")
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y_category, y_priority, y_sentiment = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_category, test_size=0.2, random_state=42
        )
        _, _, y_pri_train, y_pri_test = train_test_split(
            X, y_priority, test_size=0.2, random_state=42
        )
        _, _, y_sent_train, y_sent_test = train_test_split(
            X, y_sentiment, test_size=0.2, random_state=42
        )
        
        # Train category classifier
        print("Training category classifier...")
        self.category_model = self.build_pipeline()
        self.category_model.fit(X_train, y_cat_train)
        cat_score = self.category_model.score(X_test, y_cat_test)
        print(f"Category accuracy: {cat_score:.4f}")
        
        # Train priority classifier
        print("\nTraining priority classifier...")
        self.priority_model = self.build_pipeline()
        self.priority_model.fit(X_train, y_pri_train)
        pri_score = self.priority_model.score(X_test, y_pri_test)
        print(f"Priority accuracy: {pri_score:.4f}")
        
        # Train sentiment classifier
        print("\nTraining sentiment classifier...")
        self.sentiment_model = self.build_pipeline()
        self.sentiment_model.fit(X_train, y_sent_train)
        sent_score = self.sentiment_model.score(X_test, y_sent_test)
        print(f"Sentiment accuracy: {sent_score:.4f}")
        
        # Save models
        self.save_models()
        
        print("\n=== Training Complete ===\n")
        
        return {
            'category_accuracy': cat_score,
            'priority_accuracy': pri_score,
            'sentiment_accuracy': sent_score
        }
    
    def save_models(self):
        """Save trained models and encoders"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(f"{self.output_dir}/category_model.pkl", 'wb') as f:
            pickle.dump(self.category_model, f)
        
        with open(f"{self.output_dir}/priority_model.pkl", 'wb') as f:
            pickle.dump(self.priority_model, f)
        
        with open(f"{self.output_dir}/sentiment_model.pkl", 'wb') as f:
            pickle.dump(self.sentiment_model, f)
        
        # Save classifiers' label encoders
        with open(f"{self.output_dir}/encoders.pkl", 'wb') as f:
            pickle.dump({
                'category': self.category_encoder,
                'priority': self.priority_encoder,
                'sentiment': self.sentiment_encoder
            }, f)
        
        print(f"\nModels saved to: {self.output_dir}")
        
    
    def predict(self, text: str) -> dict:
        """Make prediction for new text"""
        if (
        self.category_model is None
        or self.priority_model is None
        or self.sentiment_model is None
        ):
            raise RuntimeError("Models are not loaded")
    
        category_idx = self.category_model.predict([text])[0]
        priority_idx = self.priority_model.predict([text])[0]
        sentiment_idx = self.sentiment_model.predict([text])[0]
        
        return {
            'category': self.category_encoder.inverse_transform([category_idx])[0],
            'priority': self.priority_encoder.inverse_transform([priority_idx])[0],
            'sentiment': self.sentiment_encoder.inverse_transform([sentiment_idx])[0]
        }


if __name__ == "__main__":
    classifier = TicketClassifier()
    results = classifier.train()
    
    # Test prediction
    test_text = "I can't access my account and need help immediately"
    prediction = classifier.predict(test_text)
    
    print(f"\nTest Prediction:")
    print(f"Text: {test_text}")
    print(f"Category: {prediction['category']}")
    print(f"Priority: {prediction['priority']}")
    print(f"Sentiment: {prediction['sentiment']}")
