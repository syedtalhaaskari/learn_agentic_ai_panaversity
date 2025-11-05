import os
import asyncio
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunConfig,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from dotenv import load_dotenv

from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str


agent = Agent(
    name="InfoCollector",
    instructions="Extract person information from the user's message.",
    output_type=PersonInfo,
)

run_config = RunConfig(model=model, tracing_disabled=True)


async def main():
    result = await Runner.run(
        agent,
        "Hi, I'm Talha, I'm 28 years old and I work as a Software Engineer.",
        run_config=run_config,
    )

    # Now you get perfect structured data!
    print("Type:", type(result.final_output))  # <class 'PersonInfo'>
    print("Name:", result.final_output.name)  # "Alice"
    print("Age:", result.final_output.age)  # 25
    print("Job:", result.final_output.occupation)  # "teacher"


if __name__ == "__main__":
    asyncio.run(main())
