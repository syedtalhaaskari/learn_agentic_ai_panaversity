import os
from dotenv import load_dotenv
import asyncio

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    MaxTurnsExceeded,
    set_tracing_disabled,
    StopAtTools,
    RunContextWrapper,
    AgentBase,
)
from dataclasses import dataclass

load_dotenv("../.env")
# set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


@dataclass
class UserScope:
    is_admin: bool


async def is_weather_allowed(
    ctx: RunContextWrapper[UserScope], agent: AgentBase[UserScope]
) -> bool:
    print("Checking if weather is allowed...", ctx.context)
    return True if ctx.context.is_admin else False


@function_tool(is_enabled=is_weather_allowed)
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"Sunny"


base_agent = Agent(
    name="WeatherAgent",
    model=model,
    tools=[get_weather],
    # tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather"]),
)


async def main():
    non_admin_scope = UserScope(is_admin=True)
    res = await Runner.run(
        base_agent, "What is weather in Lahore", context=non_admin_scope
    )
    print(res.final_output)


if __name__ == "__main__":
    asyncio.run(main())
