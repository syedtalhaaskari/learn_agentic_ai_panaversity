import os
import asyncio
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    ModelSettings,
    function_tool,
)

# 🌿 Load environment variables
load_dotenv()
# set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)

# 1) Two tiny specialists
spanish = Agent(
    name="Spanish Translator",
    instructions="Translate what the user says into Spanish. Only output Spanish.",
    model=model,
)

summarizer = Agent(
    name="Summarizer",
    instructions="Summarize the given text in 2 short bullet points.",
    model=model,
)


# 2) Wrap specialists as TOOLS
# translate_to_spanish = spanish.as_tool(
#     tool_name="translate_to_spanish",
#     tool_description="Translate user text to Spanish.",
# )
# summarize_text = summarizer.as_tool(
#     tool_name="summarize_text", tool_description="Summarize text in 2 bullets."
# )
@function_tool
async def translate_to_spanish(query: str) -> str:
    result = await Runner.run(spanish, query, max_turns=3)
    return result.final_output


@function_tool
async def summarize_text(query: str) -> str:
    result = await Runner.run(summarizer, query, max_turns=2)
    return result.final_output


# 3) Orchestrator (keeps the mic)
coach = Agent(
    name="Writing Coach",
    instructions=(
        "You help users improve messages.\n"
        "- If they say 'translate to Spanish', call translate_to_spanish.\n"
        "- If they say 'summarize', call summarize_text.\n"
        "- Otherwise, give a short tip."
    ),
    tools=[translate_to_spanish, summarize_text],
    model=model,
)


async def main():
    # Example A: ask for Spanish translation
    r1 = await Runner.run(
        coach, "Please translate to Spanish: I love learning with hands-on examples."
    )
    print("A) Final reply (from COACH, using a tool):", r1.final_output, "\n")

    # Example B: ask for a summary
    r2 = await Runner.run(
        coach,
        "Summarize: Large language models help with drafting, coding, and research.",
    )
    print("B) Final reply (from COACH, using a tool):", r2.final_output, "\n")

    # Example C: no tool needed
    r3 = await Runner.run(coach, "How can I make my email more polite?")
    print("C) Final reply (plain COACH advice):", r3.final_output)


if __name__ == "__main__":
    asyncio.run(main())
