import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    RunResult,
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

user_chat: list[dict] = []
while True:
    user_input = input("Enter your input (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    if user_input.lower() == "view":
        print("\nCurrent Chat History:", user_chat)
        continue

    user_message = {"role": "user", "content": user_input}
    user_chat.append(user_message)

    res: RunResult = Runner.run_sync(starting_agent=base_agent, input=user_chat)

    user_chat = res.to_input_list()

    print("\nAGENT RESPONSE:", res.final_output)
