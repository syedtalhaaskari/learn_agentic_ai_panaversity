import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    StopAtTools,
    set_tracing_disabled,
)

load_dotenv("../.env")
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"Sunny"


@function_tool
def get_travel_plan(city: str) -> str:
    """Plan Travel for your city"""
    return f"Travel Plan is not available"


base_agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[get_weather, get_travel_plan],
    tool_use_behavior=StopAtTools(
        # stop_at_tool_names=["get_travel_plan"]
        stop_at_tool_names=["get_travel_plan", "get_weather"]
    ),
)

res = Runner.run_sync(base_agent, "What is weather in Lahore")
# res = Runner.run_sync(base_agent, "Make me travel plan for Lahore")
print(res.final_output)
