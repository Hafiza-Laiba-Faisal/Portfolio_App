import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, handoff

_ = load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

OPENAI_API_KEY='dummy'

model="gemini-2.0-flash"
set_tracing_disabled(disabled=False)
from agents import function_tool

@function_tool
def hydro_met_tool_fn(rainfall: float, river_level: float, area_name: str) -> str:
    """Predict flood risk for a given area.

    Args:
        rainfall: Rainfall in mm.
        river_level: River level in cm.
        area_name: Name of the area being assessed.
    """
    score = (rainfall * 0.6 + river_level * 0.4) / 100
    if score > 0.7:
        risk = "HIGH"
    elif score > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return (
        f"Flood risk for {area_name}: {risk}. "
        f"(rainfall={rainfall}, river_level={river_level}, score={score:.2f})"
    )
hydro_met_agent = Agent(
    name="HydroMetAgent",
    instructions=(
        "You are a flood risk forecasting agent. "
        "Always call hydro_met_tool_fn to analyze rainfall and river data "
        "and provide a clear risk report to the coordinator."
    ),
    tools=[hydro_met_tool_fn],
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
)



# ---------------------------
# Context Model
# ---------------------------
class UserContext(BaseModel):
    scenario: str
    rainfall: float = 0
    river_level: float = 0
    evacuees: int = 0
    camp_capacity: int = 0
    houses_damaged: int = 0


# ---------------------------
# Specialized Agents
# ---------------------------
hydro_agent = Agent(
    name="HydroMetAgent",
    instructions="Analyze rainfall and river data, predict flood risk, and decide if evacuation is needed.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

evac_agent = Agent(
    name="EvacuationAgent",
    instructions="Plan evacuation routes and assign shelters for affected population clusters.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

relief_agent = Agent(
    name="ReliefAgent",
    instructions="Manage relief resources: tents, food, shortages, and coordinate with NGOs.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

recon_agent = Agent(
    name="ReconstructionAgent",
    instructions="Plan reconstruction, calculate aid needed, and propose safe return timelines.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

# ---------------------------
# Coordinator Agent (Cortex)
# ---------------------------
cortex = Agent(
    name="NighebanCortex",
    instructions=(
        "You are the national flood response coordinator. "
        "First run HydroMetAgent as a tool. Based on risk, call other tools like evacuation, relief, reconstruction."
    ),
    tools=[
        hydro_met_agent.as_tool(
            tool_name="analyze_flood_risk",
            tool_description="Analyze rainfall & river data using HydroMetAgent"
        )
    ],
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    ),
)


# Define Handoffs (delegation rules)
cortex.handoffs = [
    handoff(hydro_agent, is_enabled=lambda ctx, agent: True),  # always start here
    handoff(evac_agent, is_enabled=lambda ctx, agent: ctx.context.river_level > 70 or ctx.context.rainfall > 100),
    handoff(relief_agent, is_enabled=lambda ctx, agent: ctx.context.evacuees > ctx.context.camp_capacity),
    handoff(recon_agent, is_enabled=lambda ctx, agent: ctx.context.houses_damaged > 0),
]

# ---------------------------
# Runner
# ---------------------------
# ---------------------------
# Runner
# ---------------------------
async def main():
    context = UserContext(
        scenario="Monsoon flood alert",
        rainfall=150,
        river_level=90,
        evacuees=2000,
        camp_capacity=1200,
        houses_damaged=600,
    )

    result = await Runner.run(
        cortex,
        input="Analyze flood risk for Lahore with rainfall=150mm and river_level=90cm.",
        context=context,   # <-- yahan context dena hoga
    )

    print("\n=== Final Output ===")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
