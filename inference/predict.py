"""
Predict - Generate automated responses using RAG
"""
import multiprocessing as mp
import os
from typing import Dict, Optional
from dotenv import load_dotenv
import openai
from google import genai
from google.genai import types
from .model_loader import ModelLoader
from vector_store.faiss_index import FAISSIndex


load_dotenv()



class TicketResponseGenerator:
    """Generate automated ticket responses using RAG"""
    
    def __init__(
        self,
        llm_provider: str|None = None,
        model_name: str|None = None,
        temperature: float|None = None,
        max_tokens: int| None = None
    ):
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "gemini")
        self.model_name = model_name or "gemini-pro"

        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "500"))
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.vector_index = FAISSIndex()
        
        # Initialize LLM client
        if self.llm_provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.llm_provider == "gemini":
            self.client =genai.Client( api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def analyze_ticket(self, subject: str, description: str) -> Dict:
        """Analyze ticket and classify it"""
        # Combine subject and description
        full_text = f"{subject} {description}"
        
        # Get predictions
        predictions = self.model_loader.predict_all(full_text)
        
        # Get prediction probabilities
        probabilities = self.model_loader.get_prediction_probabilities(full_text)
        
        return {
            'category': predictions['category'],
            'priority': predictions['priority'],
            'sentiment': predictions['sentiment'],
            'probabilities': probabilities
        }
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from knowledge base"""
        return self.vector_index.get_context(query, top_k=top_k)
    
    def generate_response_gemini(
        self,
        ticket_info: Dict,
        context: str
    ) -> str:
        """Generate response using Gemini"""
        system_prompt = """You are a helpful customer support assistant. 
Your role is to provide accurate, empathetic, and professional responses to customer tickets.
Use the provided knowledge base context to inform your response.
Be concise but thorough. Always maintain a friendly and helpful tone."""
        
        user_prompt = f"""
Customer Ticket:
Subject: {ticket_info['subject']}
Description: {ticket_info['description']}

Ticket Analysis:
- Category: {ticket_info.get('category', 'Unknown')}
- Priority: {ticket_info.get('priority', 'Medium')}
- Sentiment: {ticket_info.get('sentiment', 'Neutral')}

Relevant Knowledge Base:
{context}

Please generate a helpful response to this customer ticket.
"""
        
        response = self.client.models.generate_content( # type: ignore
            model= self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=system_prompt),
                        types.Part(text=user_prompt)
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        if not response.text:
             return ""
        return response.text
    
    
    def generate_response(
        self,
        subject: str,
        description: str,
        customer_email: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: analyze ticket and generate response
        
        Args:
            subject: Ticket subject
            description: Ticket description
            customer_email: Customer email (optional)
            
        Returns:
            Dictionary with analysis and generated response
        """
        # Analyze ticket
        analysis = self.analyze_ticket(subject, description)
        
        # Get relevant context
        query = f"{subject} {description}"
        context = self.get_relevant_context(query)
        
        # Prepare ticket info
        ticket_info = {
            'subject': subject,
            'description': description,
            'customer_email': customer_email,
            **analysis
        }
        
        # Generate response based on provider
        if self.llm_provider == "gemini":
            response_text = self.generate_response_gemini(ticket_info, context)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        return {
            'ticket_analysis': {
                'category': analysis['category'],
                'priority': analysis['priority'],
                'sentiment': analysis['sentiment']
            },
            'generated_response': response_text,
            'confidence_scores': analysis['probabilities']
        }


if __name__ == "__main__":
    # Test the response generator
    generator = TicketResponseGenerator()
    
    # Example ticket
    result = generator.generate_response(
        subject="Cannot login to account",
        description="I've been trying to login but it says my password is incorrect. I tried resetting it but haven't received the email.",
        customer_email="user@example.com"
    )
    
    print("=== Ticket Analysis ===")
    print(f"Category: {result['ticket_analysis']['category']}")
    print(f"Priority: {result['ticket_analysis']['priority']}")
    print(f"Sentiment: {result['ticket_analysis']['sentiment']}")
    
    print("\n=== Generated Response ===")
    print(result['generated_response'])
