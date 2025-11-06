import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    AgentHooks,
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


class HelloAgentHooks(AgentHooks):
    def __init__(self, lifecycle_name: str):
        self.lifecycle_name = lifecycle_name

    async def on_start(self, context, agent):
        print(
            f"\n\n[{self.lifecycle_name}] Agent {agent.name} starting with context: {context}\n\n"
        )

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        print(
            f"\n\n[{self.lifecycle_name}] LLM call starting with system prompt: {system_prompt} and input items: {input_items}\n\n"
        )

    async def on_llm_end(self, context, agent, response):
        print(
            f"\n\n[{self.lifecycle_name}] LLM call ended with response: {response}\n\n"
        )

    async def on_end(self, context, agent, output):
        print(
            f"\n\n[{self.lifecycle_name}] Agent {agent.name} ended with output: {output}\n\n"
        )


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


math_agent: Agent = Agent(
    name="Math Agent",
    instructions="You are a helpful math assistant.",
    model=model,
    hooks=HelloAgentHooks("Math Agent Lifecycle"),
)


news_agent: Agent = Agent(
    name="News Agent",
    instructions="You are a helpful news assistant. Let math related questions handle by math_agent",
    model=model,
    hooks=HelloAgentHooks("News Agent Lifecycle"),
    handoffs=[math_agent],
)


base_agent: Agent = Agent(
    name="Weather Agent",
    instructions="""
    You only talk about weather.
    Let news_agent handle the news things.
    """,
    model=model,
    tools=[get_weather],
    hooks=HelloAgentHooks("Weather Agent Lifecycle"),
    handoffs=[news_agent],
)

res = Runner.run_sync(
    base_agent,
    "What's the latest news about Qwen Code - seems like it can give though time to claude code. What is 2 + 2",
)
print(res.last_agent.name)
print(res.final_output)
