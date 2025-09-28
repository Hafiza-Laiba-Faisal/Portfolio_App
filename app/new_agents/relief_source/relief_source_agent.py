import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, validator, ValidationError, field_validator
from agents import (
    Agent,                           # 🤖 Core agent class
    Runner,                          # 🏃 Runs the agent
    AsyncOpenAI,                     # 🌐 OpenAI-compatible async client
    OpenAIChatCompletionsModel,     # 🧠 Chat model interface
    function_tool,                   # 🛠️ Decorator to turn Python functions into tools
    set_default_openai_client,      # ⚙️ (Optional) Set default OpenAI client
    set_tracing_disabled,           # 🚫 Disable internal tracing/logging
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# Configure Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# 🧠 Chat model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client)
# -------------------------------
# Models with Guardrails
# -------------------------------

class Camp(BaseModel):
    name: str
    location: tuple[float, float]
    capacity: int
    current_evacs: int = 0

    @field_validator("capacity")
    def validate_capacity(cls, v):
        if v <= 0:
            raise ValueError("Camp capacity must be > 0")
        return v

    @field_validator("current_evacs")
    def validate_evacs(cls, v, info):
        if "capacity" in info.data and v > info.data["capacity"]:
            raise ValueError("Evacuees cannot exceed capacity")
        return v

    @field_validator("current_evacs")
    def validate_evacs(cls, v, values):
        if "capacity" in values and v > values["capacity"]:
            raise ValueError("Evacuees cannot exceed capacity")
        return v


class ResourceStock(BaseModel):
    food: int
    tents: int
    medical_kits: int

    @field_validator("*")
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Resource counts cannot be negative")
        return v


class ReliefInput(BaseModel):
    area_name: str
    camps: list[Camp]
    resources: ResourceStock
    new_registrations: int = 0


class ReliefOutput(BaseModel):
    updated_camps: list[dict]
    updated_resources: dict
    shortages: list[str]
    future_needs: dict
    dashboard: dict


# -------------------------------
# Relief Management Tool
# -------------------------------
class ReliefTool:
    def __init__(self):
        self.history = []  # Track daily updates

    async def validate_input(self, input_data: ReliefInput) -> bool:
        """Optional LLM-based validation"""
        try:
            input_str = input_data.json()
            prompt = (
                f"Validate relief data sanity for {input_data.area_name}.\n"
                f"Return JSON: {{'is_valid': bool, 'reason': str | null}}"
            )
            response = await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"[RELIEF INPUT VALIDATION] {result}")
            return result.get("is_valid", True)
        except Exception as e:
            logger.warning(f"Gemini validation failed: {e}")
            return True

    def update_camp_status(self, camps: list[Camp], new_registrations: int):
        """Distribute new evacuees into camps"""
        remaining = new_registrations
        updated = []
        shortages = []

        for camp in camps:
            free_space = camp.capacity - camp.current_evacs
            if free_space > 0 and remaining > 0:
                move_in = min(free_space, remaining)
                camp.current_evacs += move_in
                remaining -= move_in

            updated.append({
                "name": camp.name,
                "location": camp.location,
                "capacity": camp.capacity,
                "current_evacs": camp.current_evacs,
                "occupancy_pct": round(camp.current_evacs / camp.capacity * 100, 2),
            })

            if camp.current_evacs >= camp.capacity:
                shortages.append(f"Camp {camp.name} is at full capacity")

        if remaining > 0:
            shortages.append(f"{remaining} evacuees still need shelter")

        return updated, shortages

    def update_resources(self, resources: ResourceStock, camps: list[Camp]):
        """Update stock based on current evacuees"""
        total_evacs = sum(c.current_evacs for c in camps)

        food_needed = total_evacs * 3  # meals/day
        tents_needed = total_evacs // 4
        medical_needed = total_evacs // 50

        shortages = []
        if resources.food < food_needed:
            shortages.append("Food shortage")
        if resources.tents < tents_needed:
            shortages.append("Tent shortage")
        if resources.medical_kits < medical_needed:
            shortages.append("Medical kit shortage")

        return {
            "food": resources.food,
            "tents": resources.tents,
            "medical_kits": resources.medical_kits,
            "food_needed": food_needed,
            "tents_needed": tents_needed,
            "medical_needed": medical_needed,
        }, shortages

    def predict_future_needs(self, camps: list[Camp], resources: ResourceStock):
        """Predict demand growth based on trends"""
        total_evacs = sum(c.current_evacs for c in camps)
        growth_rate = 1.1  # Assume 10% daily growth for now
        projected = int(total_evacs * growth_rate)

        return {
            "today_evacs": total_evacs,
            "projected_tomorrow": projected,
            "extra_food_needed": (projected * 3) - resources.food,
            "extra_tents_needed": (projected // 4) - resources.tents,
            "extra_medical_needed": (projected // 50) - resources.medical_kits,
        }

    async def run(self, input_data: dict) -> ReliefOutput:
        """Main workflow with guardrails"""
        try:
            # Pydantic validation
            try:
                relief_input = ReliefInput(**input_data)
            except ValidationError as e:
                logger.error(f"Input validation error: {e}")
                raise ValueError(f"Invalid input: {e}")

            # Optional Gemini validation
            is_valid = await self.validate_input(relief_input)
            if not is_valid:
                logger.warning("⚠️ Gemini rejected the input, but proceeding with fallback...")
                # instead of raising error, we continue safely

            # Update camps
            updated_camps, camp_shortages = self.update_camp_status(
                relief_input.camps, relief_input.new_registrations
            )

            # Update resources
            updated_resources, res_shortages = self.update_resources(
                relief_input.resources, relief_input.camps
            )

            # Predict future needs
            future_needs = self.predict_future_needs(
                relief_input.camps, relief_input.resources
            )

            # Dashboard summary
            dashboard = {
                "total_camps": len(updated_camps),
                "total_evacs": sum(c["current_evacs"] for c in updated_camps),
                "capacity_filled": sum(c["current_evacs"] for c in updated_camps) / sum(c["capacity"] for c in updated_camps) * 100,
                "resource_status": updated_resources,
            }

            shortages = camp_shortages + res_shortages

            return ReliefOutput(
                updated_camps=updated_camps,
                updated_resources=updated_resources,
                shortages=shortages,
                future_needs=future_needs,
                dashboard=dashboard,
            )

        except Exception as e:
            logger.error(f"ReliefTool error: {e}")
            raise


# -------------------------------
# Agent Handler
# -------------------------------
relief_tool = ReliefTool()

async def handle_input(data: dict) -> dict:
    result = await relief_tool.run(data)
    return result.dict()



from agents import function_tool

relief_tool_instance = ReliefTool()

from pydantic import BaseModel

class ReliefInputModel(BaseModel):
    area_name: str
    camps: list[Camp]
    resources: ResourceStock
    new_registrations: int = 0

@function_tool
async def relief_management_tool(input_data: ReliefInputModel) -> ReliefOutput:
    result = await relief_tool_instance.run(input_data.dict())
    return result

from agents import Agent, Runner

# Define agent with the tool
relief_agent = Agent(
    name="ReliefAgent",
    instructions="Manage relief operations: assign evacuees, track resources, and predict future needs.",
    tools=[relief_management_tool],
    model=model  # Chat model
)

# -------------------------------
# Example Test
# -------------------------------
if __name__ == "__main__":
    sample_input = {
        "area_name": "Punjab Region",
        "camps": [
            {"name": "Camp A", "location": (31.5, 74.3), "capacity": 1000, "current_evacs": 800},
            {"name": "Camp B", "location": (31.6, 74.4), "capacity": 500, "current_evacs": 450},
        ],
        "resources": {"food": 4000, "tents": 300, "medical_kits": 50},
        "new_registrations": 400,
    }

    res = asyncio.run(handle_input(sample_input))
    print(json.dumps(res, indent=2))
