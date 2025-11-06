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

temp_session = SQLiteSession("temp_conversation")

persistent_session = SQLiteSession("user_123", "conversations.db")

agent = Agent(name="Assistant", instructions="You are helpful.", model=model)

# Use temporary session
result1 = Runner.run_sync(
    agent, "Remember: my favorite color is blue", session=temp_session
)

# Use persistent session
result2 = Runner.run_sync(
    agent, "Remember: my favorite color is blue", session=persistent_session
)

print("Both sessions now remember your favorite color!")
print("But only the persistent session will remember after restarting the program.")
