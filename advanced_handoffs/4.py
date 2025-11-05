import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    handoff,
    # set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
    RunContextWrapper,
    HandoffInputData,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from dotenv import load_dotenv

load_dotenv("../.env")
# set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


def summarized_news_transfer(data: HandoffInputData) -> HandoffInputData:
    print("\n\n[HANDOFF] Summarizing news transfer...\n\n")
    summarized_conversation = "Get latest tech news?"

    print("\n\n[ITEM 1]", data.input_history)
    print("\n\n[ITEM 2]", data.pre_handoff_items)
    print("\n\n[ITEM 1]", data.new_items)

    return HandoffInputData(
        input_history=summarized_conversation,
        pre_handoff_items=(),
        new_items=(),
    )


@function_tool
def get_weather(city: str) -> str:
    """A simple function to get the weather for a user."""
    return f"The weather for {city} is sunny."


news_agent: Agent = Agent(
    name="NewsAgent",
    instructions="You get latest news about tech community and share it with me. Always transfer back to WeatherAgent after answering the questions",
    model=model,
)

planner_agent: Agent = Agent(
    name="PlannerAgent",
    instructions="You get latest news about tech community and share it with me. Always transfer back to WeatherAgent after answering the questions",
    model=model,
)


def news_region(region: str):
    def is_news_allowed(ctx: RunContextWrapper, agent: Agent) -> bool:
        return (
            True
            if ctx.context.get("is_admin", False) and region == "us-east-1"
            else False
        )

    return is_news_allowed


weather_agent: Agent = Agent(
    name="WeatherAgent",
    instructions=f"You are weather expert - share weather updates as I travel a lot. For all Tech and News let the NewsAgent handle that part by delegation. {RECOMMENDED_PROMPT_PREFIX}",
    model=model,
    handoffs=[
        handoff(agent=news_agent, is_enabled=news_region("us-east-1")),
        planner_agent,
    ],
)

res = Runner.run_sync(
    weather_agent,
    "Check if there's any news about OpenAI after GPT-5 launch - also what's the weather SF?",
    context={"is_admin": True},
)

print("\nAGENT NAME", res.last_agent.name)
print("\n[RESPONSE:]", res.final_output)
print("\n[NEW_ITEMS:]", res.new_items)

# Now check the trace in
