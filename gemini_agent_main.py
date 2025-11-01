from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=external_client
)

agent: Agent = Agent(name="Assistant", model=llm_model)

result = Runner.run_sync(
    starting_agent=agent, input="Welcome and motivate me to learn Agentic AI"
)

print("AGENT RESPONSE: ", result.final_output)
