import os
from dotenv import load_dotenv
import asyncio

from typing import Any

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    set_tracing_disabled,
)

load_dotenv("../.env")
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


def get_weather_alternative(ctr: RunContextWrapper[Any], error: Exception) -> str:
    print("Came Here")
    return f"error_{error.__class__.__name__}"


@function_tool(description_override="", failure_error_function=get_weather_alternative)
def get_weather(city: str) -> str:
    try:
        # If Call Fails Call another service i.e get_weather_alternative
        raise ZeroDivisionError
        print("Test")
        return "Yes"
    except ValueError:
        raise ValueError("Weather service is currently unavailable.")
    except TimeoutError:
        raise TimeoutError("Weather service request timed out.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


base_agent = Agent(
    name="WeatherAgent",
    model=model,
    tools=[get_weather],
)


async def main():
    res = await Runner.run(base_agent, "What is weather in Lahore")
    print(res.final_output)


if __name__ == "__main__":
    asyncio.run(main())
