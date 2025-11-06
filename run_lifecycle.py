import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    RunHooks,
    RunContextWrapper,
)

from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


class HelloRunHooks(RunHooks):
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent):
        print(
            f"\n\n[RunLifecycle] Agent {agent.name} start with context: {context}\n\n"
        )

    async def on_llm_start(
        self, context: RunContextWrapper, agent: Agent, system_prompt, input_items
    ):
        print(
            f"\n\n[RunLifecycle] LLM call for agent {agent.name} starting with system prompt: {system_prompt} and input items: {input_items}\n\n"
        )


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


news_agent: Agent = Agent(
    name="NewsAgent",
    instructions="You are a helpful news assistant.",
    model=model,
)


base_agent: Agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant. Talk about weather and let news_agent handle the news things",
    model=model,
    tools=[get_weather],
    handoffs=[news_agent],
)

res = Runner.run_sync(
    starting_agent=base_agent,
    input="What's the latest news about Qwen Code - seems like it can give though time to claude code.",
    hooks=HelloRunHooks(),
)
print(res.last_agent.name)
print(res.final_output)
