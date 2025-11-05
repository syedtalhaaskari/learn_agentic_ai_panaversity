import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    handoff,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
)
from pydantic import BaseModel
import asyncio

from dotenv import load_dotenv

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


agent = Agent(
    name="Assistant",
    instructions="You only respond for the user's request and delegate to the expert agent if needed.",
    model=model,
)

expert_agent = Agent(
    name="Expert",
    instructions="You are an expert in the field of recursion in programming.",
    model=model,
)

agent.handoffs = [
    handoff(expert_agent, is_enabled=lambda ctx, agent: ctx.context.has_permission)
]


async def main():
    context = UserContext(
        user_id="123", subscription_tier="premium", has_permission=True
    )

    result = await Runner.run(
        agent,
        "Call the expert agent and ask about recursion in programming",
        context=context,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
