"""
Evaluation Script - Evaluate trained models
"""
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """Evaluate trained classification models"""
    
    def __init__(
        self,
        data_path: str = "data/raw/training_tickets.csv",
        models_dir: str = "models",
        output_dir: str = "evaluation_results"
    ):
        self.data_path = data_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load trained models and encoders"""
        print("Loading models...")
        
        with open(f"{self.models_dir}/category_model.pkl", 'rb') as f:
            self.category_model = pickle.load(f)
        
        with open(f"{self.models_dir}/priority_model.pkl", 'rb') as f:
            self.priority_model = pickle.load(f)
        
        with open(f"{self.models_dir}/sentiment_model.pkl", 'rb') as f:
            self.sentiment_model = pickle.load(f)
        
        with open(f"{self.models_dir}/encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            self.category_encoder = encoders['category']
            self.priority_encoder = encoders['priority']
            self.sentiment_encoder = encoders['sentiment']
        
        print("Models loaded successfully")
    
    def load_data(self):
        """Load test data"""
        df = pd.read_csv(self.data_path)
        X = df["subject"].str.cat(df["description"], sep=" ")

        
        y_category = self.category_encoder.transform(df['category'])
        y_priority = self.priority_encoder.transform(df['priority'])
        y_sentiment = self.sentiment_encoder.transform(df['sentiment'])
        
        # Use same split as training
        X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test, y_sent_train, y_sent_test = train_test_split(
            X,
            y_category,
            y_priority, 
            y_sentiment, 
            test_size=0.3, 
            random_state=42, 
            stratify=y_category
        )
       
        return X_test, y_cat_test, y_pri_test, y_sent_test
    
    def evaluate_model(self, model, X_test, y_test, encoder, name):
        """Evaluate a single model"""
        print(f"\n=== Evaluating {name} Model ===\n")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        report = classification_report(
            y_test,
            y_pred,
            labels=unique_labels,
            target_names=[encoder.classes_[i] for i in unique_labels],
            zero_division=0
        )
        print(f"\nClassification Report:\n{report}")
        
        # Save report
        with open(f"{self.output_dir}/{name}_report.txt", 'w') as f:
            f.write(f"{name} Model Evaluation\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            
            if isinstance(report, dict):
                report = json.dumps(report, indent=2)
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[encoder.classes_[i] for i in unique_labels],
            yticklabels=[encoder.classes_[i] for i in unique_labels]
        )
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{name}_confusion_matrix.png")
        plt.close()
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    def evaluate_all(self):
        """Evaluate all models"""
        print("\n=== Starting Model Evaluation ===\n")
        
        # Load test data
        X_test, y_cat_test, y_pri_test, y_sent_test = self.load_data()
        
        results = {}
        
        # Evaluate category model
        results['category'] = self.evaluate_model(
            self.category_model,
            X_test,
            y_cat_test,
            self.category_encoder,
            'Category'
        )
        
        # Evaluate priority model
        results['priority'] = self.evaluate_model(
            self.priority_model,
            X_test,
            y_pri_test,
            self.priority_encoder,
            'Priority'
        )
        
        # Evaluate sentiment model
        results['sentiment'] = self.evaluate_model(
            self.sentiment_model,
            X_test,
            y_sent_test,
            self.sentiment_encoder,
            'Sentiment'
        )
        
        # Summary
        print("\n=== Evaluation Summary ===\n")
        print(f"Category Accuracy: {results['category']['accuracy']:.4f}")
        print(f"Priority Accuracy: {results['priority']['accuracy']:.4f}")
        print(f"Sentiment Accuracy: {results['sentiment']['accuracy']:.4f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return results


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all()
