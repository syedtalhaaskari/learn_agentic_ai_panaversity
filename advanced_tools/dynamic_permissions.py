import os
from dotenv import load_dotenv
import asyncio

from pydantic import BaseModel
from typing import Any

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    set_tracing_disabled,
    AgentBase,
)

load_dotenv("../.env")
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


class UserContext(BaseModel):
    user_id: str
    subscription_tier: str = "free"  # free, premium, enterprise
    has_permission: bool = False


def premium_feature_enabled(ctx: RunContextWrapper, agent: AgentBase) -> bool:
    print(f"premium_feature_enabled()")
    print(
        ctx.context.subscription_tier,
        ctx.context.subscription_tier in ["premium", "enterprise"],
    )
    return ctx.context.subscription_tier in ["premium", "enterprise"]


@function_tool(is_enabled=premium_feature_enabled)
def get_weather(city: str) -> str:
    print(f"[ADV] get_weather()")
    return "Weather is sunny"


agent = Agent(
    name="Assistant",
    instructions="You only respond in haikus.",
    model=model,
    tools=[get_weather],
)


async def main():
    # context = UserContext(
    #     user_id="123", subscription_tier="premium", has_permission=True
    # )
    context = UserContext(
        user_id="123", subscription_tier="basic", has_permission=False
    )

    result = await Runner.run(
        agent,
        "Call the get_weather tool with city 'London'",
        context=context,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
