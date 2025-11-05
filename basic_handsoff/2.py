import os
import asyncio
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    handoff,
    RunConfig,
)

# 🌿 Load environment variables
load_dotenv("../.env")
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)

run_config = RunConfig(
    model=model,
    # model_provider=external_client,
)

# 1) Two specialists that will OWN the conversation after transfer
billing_agent = Agent(
    name="Billing Agent",
    instructions="Resolve billing problems end-to-end. Ask for any details you need.",
)

refunds_agent = Agent(
    name="Refunds Agent",
    instructions="Handle refunds end-to-end. Ask for order ID and explain next steps.",
)

# 2) Triage agent that decides WHO should take over
triage = Agent(
    name="Triage Agent",
    instructions=(
        "Greet the user and decide where to send them:\n"
        "- If the user asks about a double charge, invoice, payment, etc., hand off to Billing Agent.\n"
        "- If the user asks about refund status or returning an item, hand off to Refunds Agent.\n"
        "Once handed off, the specialist should continue the conversation."
    ),
    handoffs=[billing_agent, handoff(refunds_agent)],
)


async def main():
    r1 = await Runner.run(
        triage,
        "Hi, I returned my headset last week. What's my refund status?",
        run_config=run_config,
    )
    print("A) Final reply (from REFUNDS specialist):", r1.final_output, "\n")

    r2 = await Runner.run(
        triage, "My card was charged twice for the same order.", run_config=run_config
    )
    print("B) Final reply (from BILLING specialist):", r2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
