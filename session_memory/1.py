import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    AgentHooks,
    SQLiteSession,
)
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv("../.env")
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Be friendly and remember our conversation.",
    model=model,
)

# Create session memory
session = SQLiteSession("my_first_conversation")

print("=== First Conversation with Memory ===")

result1 = Runner.run_sync(
    agent, "Hi! My name is Talha and I love pizza.", session=session
)
print("Agent:", result1.final_output)

# Turn 2 - Agent should remember your name!
result2 = Runner.run_sync(agent, "What's my name?", session=session)
print("Agent:", result2.final_output)  # Should say "Talha"!

# Turn 3 - Agent should remember you love pizza!
result3 = Runner.run_sync(agent, "What food do I like?", session=session)

print("Agent:", result3.final_output)  # Should say "Talha"!

print("\n\nNO SESSION MEMORY\n\n")
result4 = Runner.run_sync(agent, "What's my name and what do I like?")
print("Agent:", result4.final_output)  # Should mention pizza!
