import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(disabled=True)


async def main():
    haiku_agent = Agent(
        name="Asistant",
        instructions="You only respond in haikus.",
        model=OpenAIChatCompletionsModel(
            model="gemini-2.0-flash", openai_client=client
        ),
    )

    result = await Runner.run(haiku_agent, "Tell me about recursion in programming.")

    print(result.final_output)


print(__name__)
if __name__ == "__main__":
    asyncio.run(main())
