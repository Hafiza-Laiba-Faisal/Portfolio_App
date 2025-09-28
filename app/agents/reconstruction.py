import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError, field_validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# Gemini Client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# -------------------------------
# Models with Guardrails
# -------------------------------

class AreaDamage(BaseModel):
    area_name: str
    damage_level: str  # e.g., "low", "medium", "high", "severe"
    population_affected: int
    safe_water: bool
    electricity: bool

    @field_validator("damage_level")
    def validate_damage(cls, v):
        if v not in ["low", "medium", "high", "severe"]:
            raise ValueError("damage_level must be one of: low, medium, high, severe")
        return v

    @field_validator("population_affected")
    def validate_population(cls, v):
        if v < 0:
            raise ValueError("Population cannot be negative")
        return v


class AidResources(BaseModel):
    housing_kits: int
    financial_aid: int  # in USD or PKR
    volunteers: int

    @field_validator("*")
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Values cannot be negative")
        return v


class ReconstructionInput(BaseModel):
    area: AreaDamage
    aid: AidResources
    drone_verified: bool = False
    satellite_verified: bool = False
    field_verified: bool = False


class ReconstructionOutput(BaseModel):
    safe_return_timeline: str
    recommended_actions: list[str]
    updated_aid: dict
    rehab_progress: dict
    coordination_notes: list[str]


# -------------------------------
# Reconstruction Tool
# -------------------------------
class ReconstructionTool:
    async def assess_damage(self, area: AreaDamage) -> str:
        """Simple rule-based safe return timeline"""
        if area.damage_level == "low":
            return "Return possible within 1 week"
        elif area.damage_level == "medium":
            return "Return possible in 2–3 weeks"
        elif area.damage_level == "high":
            return "Return possible in 1–2 months"
        else:
            return "Return delayed until major reconstruction (3+ months)"

    def recommend_actions(self, area: AreaDamage, aid: AidResources) -> list[str]:
        actions = []
        if not area.safe_water:
            actions.append("Provide clean water supply")
        if not area.electricity:
            actions.append("Restore electricity")
        if aid.housing_kits < (area.population_affected // 5):
            actions.append("Increase housing kits allocation")
        if aid.financial_aid < (area.population_affected * 100):
            actions.append("Expand financial aid coverage")
        return actions

    def track_rehab_progress(self, area: AreaDamage, aid: AidResources) -> dict:
        progress = {
            "water_restored": area.safe_water,
            "electricity_restored": area.electricity,
            "housing_support_pct": round(min(100, (aid.housing_kits * 5) / max(1, area.population_affected) * 100), 2),
            "financial_coverage_pct": round(min(100, aid.financial_aid / max(1, area.population_affected * 100) * 100), 2),
        }
        return progress

    def coord_notes(self, input_data: ReconstructionInput) -> list[str]:
        notes = []
        if not input_data.drone_verified:
            notes.append("Request drone survey confirmation")
        if not input_data.satellite_verified:
            notes.append("Request satellite verification")
        if not input_data.field_verified:
            notes.append("Request field survey verification")
        notes.append("Coordinate with local councils & housing authorities")
        notes.append("Sync with NDMA/PDMA rehabilitation plans")
        return notes

    async def run(self, input_data: dict) -> ReconstructionOutput:
        try:
            # Schema Guardrails
            reconstruction_input = ReconstructionInput(**input_data)

            # Assess return timeline
            timeline = await self.assess_damage(reconstruction_input.area)

            # Recommend actions
            actions = self.recommend_actions(reconstruction_input.area, reconstruction_input.aid)

            # Rehab progress
            progress = self.track_rehab_progress(reconstruction_input.area, reconstruction_input.aid)

            # Coordination
            notes = self.coord_notes(reconstruction_input)

            return ReconstructionOutput(
                safe_return_timeline=timeline,
                recommended_actions=actions,
                updated_aid=reconstruction_input.aid.model_dump(),
                rehab_progress=progress,
                coordination_notes=notes,
            )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise ValueError(f"Invalid input: {e}")


# -------------------------------
# Example Run
# -------------------------------
async def main():
    sample_input = {
        "area": {
            "area_name": "Sindh District",
            "damage_level": "high",
            "population_affected": 1200,
            "safe_water": False,
            "electricity": False,
        },
        "aid": {"housing_kits": 150, "financial_aid": 50000, "volunteers": 30},
        "drone_verified": True,
        "satellite_verified": False,
        "field_verified": False,
    }

    tool = ReconstructionTool()
    result = await tool.run(sample_input)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
