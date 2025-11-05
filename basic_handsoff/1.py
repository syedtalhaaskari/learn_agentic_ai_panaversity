import os

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled,
)
from dotenv import load_dotenv

# 🌿 Load environment variables
load_dotenv("../.env")
# set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)


@function_tool
def get_unread_whatsapp_messages() -> str:
    """Check and returns unread WhatsApp messages and share them."""
    # Simulated unread messages
    return "You have 1 unread messages: 'Create a promotional copy for our new sneakers launch!'"


# Pattern: Agent as tool
whatsapp_monitor: Agent = Agent(
    name="WhatsApp Monitor",
    instructions="""Check unread WhatsApp messages and delegate when intent indicates.
    Use tools for whatsapp interaction. Do not assume anything.
    """,
    model=model,
    tools=[get_unread_whatsapp_messages],
    handoff_description="Check WhatsApp messages.",
)

copywriter: Agent = Agent(
    name="Copywriter",
    instructions="""Create promotional copies based on WhatsApp message requests.""",
    model=model,
    handoffs=[whatsapp_monitor],
    handoff_description="Handles requests for promotional copywriting.",
)

whatsapp_monitor.handoffs = [copywriter]

result = Runner.run_sync(
    starting_agent=copywriter,
    input="""Check the WhatsApp messages and create promotional copies as needed.""",
)

print("\nACTIVE AGENT: ", result.last_agent.name)
print("\nAGENT RESPONSE: ", result.final_output)
