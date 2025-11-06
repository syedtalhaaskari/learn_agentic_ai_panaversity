import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    handoff,
    RunConfig,
    input_guardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
    output_guardrail,
    OutputGuardrailTripwireTriggered,
)

from pydantic import BaseModel

# 🌿 Load environment variables
load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or ""
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or ""

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=external_client)

run_config = RunConfig(model=model, tracing_disabled=True)


class WeatherSanitizer(BaseModel):
    weather_related: bool
    reason: str | None = None


weather_sanitizer = Agent(
    name="Weather Sanitizer",
    instructions="Check if this is a weather related query",
    output_type=WeatherSanitizer,
)

weather_output_sanitizer = Agent(
    name="Weather Output Sanitizer",
    instructions="Check if the output has only weather related answer with no sensitive or irrelevant data",
    output_type=WeatherSanitizer,
)


@input_guardrail
async def weather_input_checker(
    ctx: RunContextWrapper, agent: Agent, input_data
) -> GuardrailFunctionOutput:
    result = await Runner.run(weather_sanitizer, input_data, run_config=run_config)
    print("\n[WEATHER SANITIZER RESPONSE]", result.final_output)

    return GuardrailFunctionOutput(
        output_info="passed",
        tripwire_triggered=result.final_output.weather_related is False,
    )


@output_guardrail
async def weather_response_checker(
    ctx: RunContextWrapper, agent: Agent, output
) -> GuardrailFunctionOutput:
    result = await Runner.run(weather_output_sanitizer, output, run_config=run_config)
    print("Output Guardrail Input:", output)
    print("Output Guardrail response:", result.final_output)
    return GuardrailFunctionOutput(
        output_info="passed",
        tripwire_triggered=result.final_output.weather_related is False,
    )


base_agent: Agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful assistant.",
    input_guardrails=[weather_input_checker],
    output_guardrails=[weather_response_checker],
)

try:
    res = Runner.run_sync(
        base_agent,
        [{"role": "user", "content": "What's the weather like in SF?"}],
        run_config=run_config,
    )

    print("[OUTPUT]", res.to_input_list())
except InputGuardrailTripwireTriggered:
    print("Alert: Guardrail input tripwire was triggered!")
except OutputGuardrailTripwireTriggered:
    print("Alert: Guardrail output tripwire was triggered!")
