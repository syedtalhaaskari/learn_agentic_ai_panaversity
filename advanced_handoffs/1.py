import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    handoff,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    RunContextWrapper,
)

from dotenv import load_dotenv

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
    return f"The weather for {city} is sunny."


def on_handoff(ctx: RunContextWrapper):
    print("News agent called")


news_agent: Agent = Agent(
    name="NewsAgent",
    instructions="You get latest news about tech community and share it with me.",
    model=model,
    tools=[get_weather],
)

weather_agent = Agent(
    name="WeatherAgent",
    instructions="You are weather expert - share weather updates as I travel a lot. For all Tech and News let the NewsAgent handle that part by delegation.",
    model=model,
    tools=[get_weather],
    handoffs=[handoff(agent=news_agent, on_handoff=on_handoff)],
)

res = Runner.run_sync(
    weather_agent, "Check if there's any news about OpenAI after GPT-5 launch?"
)
print("\nAGENT NAME", res.last_agent.name)
print("\n[RESPONSE:]", res.final_output)

# Now check the trace in
