import os

# from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

assistant_agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(assistant_agent, "Hello, how are you?", run_config=config)

print(result.final_output)
