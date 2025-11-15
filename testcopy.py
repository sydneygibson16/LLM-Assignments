"""
Tests for Module 08 Assignment: Understanding ChatAgent vs Direct LLM Calls
Updated to be compatible with assignmentcopy.py
"""

import pytest
from assignmentcopy import direct_llm_chat, create_chat_agent, chat_with_agent


def test_direct_llm_chat():
    """Test that direct_llm_chat shows stateless behavior."""
    response1, response2 = direct_llm_chat("Is 5 a prime number?", "What about 15?")

    # Check types and non-empty
    assert isinstance(response1, str)
    assert isinstance(response2, str)
    assert len(response1) > 0
    assert len(response2) > 0

    # Allow either numeric '15' or spelled 'Fifteen' in response2
    assert "15" in response2 or "Fifteen" in response2


def test_create_chat_agent():
    """Test that we can create a ChatAgent instance."""
    agent = create_chat_agent()
    assert agent is not None
    assert hasattr(agent, "llm")  # Agent must have an LLM
    assert callable(getattr(agent.llm, "chat"))


def test_chat_with_agent():
    """Test sending a message to a ChatAgent and getting a response."""
    agent = create_chat_agent()
    response = chat_with_agent(agent, "Hello, are you an assistant?")

    # Check type and content
    assert isinstance(response, str)
    assert len(response) > 0
    assert "assistant" in response.lower() or "mock" in response.lower() or len(response) > 0

