import os
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    handoff,
    RunContextWrapper,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
    function_tool,
)
from agents.extensions import handoff_filters
from pydantic import BaseModel
import asyncio

from dotenv import load_dotenv

load_dotenv("../.env")
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


# --- Define the data for our "briefing note" ---
class HandoffData(BaseModel):
    summary: str


# --- Define our specialist agents ---
billing_agent = Agent(
    name="Billing Agent", instructions="Handle billing questions.", model=model
)
technical_agent = Agent(
    name="Technical Support Agent",
    instructions="Troubleshoot technical issues.",
    model=model,
)


# --- Define our on_handoff callback ---
def log_the_handoff(ctx: RunContextWrapper, input_data: HandoffData):
    print(f"\n[SYSTEM: Handoff initiated. Briefing: '{input_data.summary}']\n")


# --- TODO 1: Create the advanced handoffs ---

# Create a handoff to `billing_agent`.
# - Override the tool name to be "transfer_to_billing".
# - Use the `log_the_handoff` callback.
# - Require `HandoffData` as input.
to_billing_handoff = handoff(
    agent=billing_agent, on_handoff=log_the_handoff, input_type=HandoffData
)

# Create a handoff to `technical_agent`.
# - Use the `log_the_handoff` callback.
# - Require `HandoffData` as input.
# - Add an input filter: `handoff_filters.remove_all_tools`.
to_technical_handoff = handoff(
    agent=technical_agent, on_handoff=log_the_handoff, input_type=HandoffData
)


@function_tool
def diagnose():
    print("Diagnosed Successfully")


# --- Triage Agent uses the handoffs ---
triage_agent = Agent(
    name="Triage Agent",
    instructions="First, use the 'diagnose' tool. Then, based on the issue, handoff to the correct specialist with a summary.",
    tools=[
        # A dummy tool for the triage agent to use
        diagnose
    ],
    handoffs=[to_billing_handoff, to_technical_handoff],
    model=model,
)


async def main():
    print("--- Running Scenario: Billing Issue ---")
    result = await Runner.run(triage_agent, "My payment won't go through.")
    print(f"Final Reply From: {result.last_agent.name}")
    print(f"Final Message: {result.final_output}")


asyncio.run(main())
