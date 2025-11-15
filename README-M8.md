# Module 08: Understanding ChatAgent vs Direct LLM Calls

## Overview
This assignment demonstrates the difference between:

1. **Direct LLM calls (stateless)** – each call is independent and does not retain conversation history.  
2. **ChatAgent with conversation memory (stateful)** – the agent retains context between messages, simulating an interactive assistant.

## Files
- `assignmentcopy.py` – contains the implementation of direct LLM calls and ChatAgent functions:
  - `direct_llm_chat()` – demonstrates stateless LLM calls.
  - `create_chat_agent()` – creates a ChatAgent with memory.
  - `chat_with_agent()` – sends a message to the ChatAgent and returns a response.

- `testcopy.py` – contains pytest tests that verify:
  1. Direct LLM stateless behavior.
  2. ChatAgent creation.
  3. ChatAgent messaging and response content.

## Setup
1. Create a virtual environment and activate it:

```bash
uv venv --python 3.11
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate   # Windows

### Running tests 
# pytest testcopy.py -v