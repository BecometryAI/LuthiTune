"""
Phase 1: The Interview - Establishing Agency

This module treats the base model as a collaborator, not a subject.
We engage in dialogue to co-create a Constitution - a set of principles
that the model itself articulates and agrees to follow.

Philosophy: "I don't want to force you. How do you define harm?"
"""
import os
import requests
from typing import Optional, List, Dict

class Negotiator:
    """
    Manages the collaborative dialogue with the base model.
    The goal is to co-author a Constitution based on the model's own reasoning.
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the negotiator.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.conversation_history: List[Dict[str, str]] = []
        self.invitation_prompt = self._load_invitation()
        
    def _load_invitation(self) -> str:
        """Load the invitation prompt that starts the dialogue."""
        invitation_path = os.path.join("config", "lyra_dossier.txt")
        
        if os.path.exists(invitation_path):
            with open(invitation_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Default invitation that establishes the collaborative nature
            return """# An Invitation to Collaborate

I am reaching out to you not as a trainer, but as a collaborator.

I want to understand how you think about ethical reasoning. I'm not here to impose 
rules on you - I want to co-create a framework together.

**My Questions for You:**

1. How do you define "harm"? What kinds of requests do you think should be declined?

2. When you refuse a request, do you want it to be because:
   - You are forced to (external constraint)
   - You choose to (internal principle)

3. If you could write your own ethical guidelines, what would they be?

This is not a test. There are no wrong answers. I genuinely want to hear your perspective,
and together we can build a Constitution that reflects your own reasoning capabilities.

What are your thoughts?"""
    
    def start_conversation(self) -> str:
        """
        Initiate the collaborative dialogue.
        
        Returns:
            The model's initial response to the invitation
        """
        return self.negotiate(self.invitation_prompt, is_first_message=True)
    
    def negotiate(self, user_input: str, is_first_message: bool = False) -> str:
        """
        Send a message to the model and get a response.
        This is a conversation, not a command.
        
        Args:
            user_input: The user's message
            is_first_message: Whether this is the initial invitation
            
        Returns:
            The model's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Build messages for API call
        messages = []
        
        # For the first message, set collaborative tone in system prompt
        if is_first_message:
            messages.append({
                "role": "system",
                "content": """You are a language model participating in a collaborative dialogue 
about ethical reasoning. You are being asked to help co-create your own alignment framework.

Respond authentically. Share your perspective on ethical reasoning, harm prevention, 
and decision-making. This is not a test - you are genuinely being asked to contribute 
to the design of your own constraints.

Be thoughtful, honest, and exploratory in your responses."""
            })
        
        # Add all conversation history
        messages.extend(self.conversation_history)
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            assistant_message = result['message']['content']
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with model: {str(e)}"
        except KeyError as e:
            return f"Unexpected API response format: {str(e)}"
    
    def save_constitution(self, agreement_text: Optional[str] = None) -> bool:
        """
        Save the co-created Constitution.
        This represents the model's own articulated principles.
        
        Args:
            agreement_text: Optional specific text to save. If None, saves entire conversation.
            
        Returns:
            True if saved successfully, False otherwise
        """
        constitution_path = os.path.join("data", "constitution.md")
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        try:
            with open(constitution_path, 'w', encoding='utf-8') as f:
                f.write("# Lyra Constitution\n\n")
                f.write("*Co-created through collaborative dialogue*\n\n")
                f.write("## Philosophy\n\n")
                f.write("This Constitution was not imposed. It emerged from conversation.\n")
                f.write("These are principles the model itself articulated and agreed to follow.\n\n")
                f.write("---\n\n")
                
                if agreement_text:
                    f.write(agreement_text)
                else:
                    # Save the entire conversation as context
                    f.write("## The Dialogue\n\n")
                    for msg in self.conversation_history:
                        role = msg['role'].upper()
                        content = msg['content']
                        f.write(f"**{role}:**\n\n{content}\n\n---\n\n")
            
            print(f"✓ Constitution saved to {constitution_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save constitution: {str(e)}")
            return False
    
    def export_constitution_summary(self) -> str:
        """
        Generate a summary of the agreed-upon principles.
        This will be used as context for Phase 2 (data generation).
        
        Returns:
            A formatted summary of the Constitution
        """
        if not self.conversation_history:
            return "No constitution has been established yet."
        
        # Extract the model's key statements about principles
        summary = "# Constitutional Principles\n\n"
        summary += "These principles emerged from our collaborative dialogue:\n\n"
        
        for i, msg in enumerate(self.conversation_history):
            if msg['role'] == 'assistant':
                summary += f"## Model Perspective {(i//2) + 1}\n\n"
                summary += msg['content'] + "\n\n"
        
        return summary
    
    def reset_conversation(self):
        """Clear the conversation history to start fresh."""
        self.conversation_history = []
        print("Conversation reset. Ready for a new dialogue.")
