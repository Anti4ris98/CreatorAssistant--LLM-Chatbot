"""Chat handling utilities for inference."""

from typing import List, Dict
from datetime import datetime


class ChatHistory:
    """Manages conversation history."""
    
    def __init__(self, max_history: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_history * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_context(self, num_turns: int = 3) -> str:
        """Get recent conversation context."""
        recent_messages = self.messages[-(num_turns * 2):]
        
        context_parts = []
        for msg in recent_messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []


def format_prompt_with_context(prompt: str, chat_history: ChatHistory, use_context: bool = True) -> str:
    """Format prompt with optional conversation context."""
    if use_context and chat_history.messages:
        context = chat_history.get_context(num_turns=2)
        formatted_prompt = f"Previous conversation:\n{context}\n\nCurrent question: {prompt}"
    else:
        formatted_prompt = prompt
    
    return formatted_prompt


def validate_input(prompt: str, max_length: int = 500) -> tuple[bool, str]:
    """Validate user input."""
    if not prompt or not prompt.strip():
        return False, "Empty prompt provided"
    
    if len(prompt) > max_length:
        return False, f"Prompt too long (max {max_length} characters)"
    
    return True, ""
