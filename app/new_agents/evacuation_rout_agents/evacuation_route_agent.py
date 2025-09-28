import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator, ValidationError
import networkx as nx
from scipy.spatial.distance import cdist
import folium

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

# Configure OpenAI client for Gemini
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# 🧠 Chat model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)
# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj == float("inf"):
            return "Infinity"
        return super().default(obj)

# -------------------------------
# Models with Guardrails
# -------------------------------
class EvacuationInput(BaseModel):
    area_name: str
    bbox: list[float]  # [min_lat, min_lon, max_lat, max_lon]
    population_clusters: list[dict] | None = None
    flood_zones: list[dict] | None = None
    transport_resources: list[dict] | None = None
    shelters: list[dict] | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 coordinates: [min_lat, min_lon, max_lat, max_lon]")
        min_lat, min_lon, max_lat, max_lon = v
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if min_lat >= max_lat or min_lon >= max_lon:
            raise ValueError("min coordinates must be smaller than max coordinates")
        return v

    @field_validator("population_clusters", mode="before")
    @classmethod
    def validate_clusters(cls, v):
        if v is None:
            return v
        for item in v:
            if "lat" not in item or "lon" not in item or "population" not in item or "vulnerability" not in item:
                raise ValueError("Each cluster must have lat, lon, population, and vulnerability")
            if not (-90 <= item["lat"] <= 90 and -180 <= item["lon"] <= 180):
                raise ValueError("Cluster coordinates out of bounds")
            if item["population"] <= 0:
                raise ValueError("Population must be positive")
            if item["vulnerability"] not in ["low", "medium", "high"]:
                raise ValueError("Vulnerability must be low, medium, or high")
        return v

    @field_validator("flood_zones", mode="before")
    @classmethod
    def validate_flood_zones(cls, v):
        if v is None:
            return v
        for item in v:
            if "lat" not in item or "lon" not in item or "radius_km" not in item or "severity" not in item:
                raise ValueError("Each flood zone must have lat, lon, radius_km, and severity")
            if not (-90 <= item["lat"] <= 90 and -180 <= item["lon"] <= 180):
                raise ValueError("Flood zone coordinates out of bounds")
            if item["radius_km"] <= 0:
                raise ValueError("Flood zone radius must be positive")
            if item["severity"] not in ["low", "medium", "high"]:
                raise ValueError("Severity must be low, medium, or high")
        return v

class EvacuationOutput(BaseModel):
    evacuation_plan: dict
    route_optimization: dict
    shelter_assignment: dict
    meeting_notices: list[dict]
    action_plan: dict
    visualization_files: list[str]
    gis_map: str

# -------------------------------
# Evacuation Tool
# -------------------------------


class EvacuationTool:
    """
    Tool for generating evacuation plans, optimizing routes,
    assigning shelters/transport, and coordinating stakeholders.
    """
    def __init__(self):
        self.graph = None
        self.email_statuses = {}

    async def validate_input(self, input_data: EvacuationInput) -> bool:
        """Use both pydantic and Gemini for validation (mocked for testing)"""
        try:
            input_str = json.dumps(input_data.model_dump())
            prompt = (
                f"Double-check input sanity for evacuation. Input: {input_str}\n"
                f"Return JSON: {{'is_valid': bool, 'reason': str | null}}"
            )
            response = await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"[EVACUATION INPUT VALIDATION] {result}")
            return result.get('is_valid', True)
        except Exception as e:
            logger.warning(f"Gemini validation failed, defaulting to True: {e}")
            return True  # Mocked response for testing

    def fetch_gis_data(self, bbox: list[float]) -> dict:
        """Fetch GIS data (mock implementation)"""
        min_lat, min_lon, max_lat, max_lon = bbox
        roads = [
            {"start": [min_lat + 0.01, min_lon + 0.01], "end": [min_lat + 0.02, min_lon + 0.02], "length_km": 1.2, "traffic_level": "low"},
            {"start": [min_lat + 0.02, min_lon + 0.02], "end": [max_lat - 0.01, max_lon - 0.01], "length_km": 5.5, "traffic_level": "medium"},
            {"start": [max_lat - 0.01, max_lon - 0.01], "end": [max_lat - 0.02, max_lon - 0.02], "length_km": 2.1, "traffic_level": "high"}
        ]
        terrain = {
            "elevation_profile": {"min_elev": 200, "max_elev": 800, "avg_slope": 0.05},
            "flood_prone_areas": [{"lat": min_lat + 0.015, "lon": min_lon + 0.015, "risk": "high"}]
        }
        return {"roads": roads, "terrain": terrain}

    def identify_population_clusters(self, input_data: EvacuationInput) -> list[dict]:
        """Identify population clusters"""
        if input_data.population_clusters:
            clusters = input_data.population_clusters
        else:
            center_lat = (input_data.bbox[0] + input_data.bbox[2]) / 2
            center_lon = (input_data.bbox[1] + input_data.bbox[3]) / 2
            clusters = [
                {"lat": center_lat - 0.01, "lon": center_lon - 0.01, "population": 2500, "vulnerability": "high"},
                {"lat": center_lat + 0.01, "lon": center_lon + 0.01, "population": 1800, "vulnerability": "medium"},
                {"lat": center_lat, "lon": center_lon, "population": 3200, "vulnerability": "low"}
            ]
        for cluster in clusters:
            cluster["evac_priority"] = cluster.get("population", 0) * (
                2 if cluster.get("vulnerability") == "high" else 1.5 if cluster.get("vulnerability") == "medium" else 1
            )
        return clusters

    def optimize_evacuation_routes(self, clusters: list[dict], shelters: list[dict], gis_data: dict) -> dict:
        """Optimize evacuation routes"""
        G = nx.Graph()
        for road in gis_data["roads"]:
            G.add_edge(
                (road["start"][0], road["start"][1]),
                (road["end"][0], road["end"][1]),
                weight=road["length_km"] * (2 if road["traffic_level"] == "high" else 1.5 if road["traffic_level"] == "medium" else 1)
            )

        def connect_point_to_graph(point):
            """Helper: connect arbitrary point to nearest road node"""
            if point not in G:
                nearest_node = min(G.nodes, key=lambda n: cdist([point], [n])[0][0])
                dist = cdist([point], [nearest_node])[0][0]
                G.add_edge(point, nearest_node, weight=dist + 0.001)

        routes = {}
        for cluster in clusters:
            cluster_point = (cluster["lat"], cluster["lon"])
            cluster_key = f"{cluster['lat']},{cluster['lon']}"
            connect_point_to_graph(cluster_point)

            best_shelter = min(shelters, key=lambda s: cdist([cluster_point], [(s["location"][0], s["location"][1])])[0][0])
            best_shelter_point = (best_shelter["location"][0], best_shelter["location"][1])
            connect_point_to_graph(best_shelter_point)

            try:
                path = nx.shortest_path(G, cluster_point, best_shelter_point, weight="weight")
                routes[cluster_key] = {
                    "to_shelter": best_shelter["name"],
                    "distance_km": cdist([cluster_point], [best_shelter_point])[0][0],
                    "estimated_time_hours": cdist([cluster_point], [best_shelter_point])[0][0] / 50,
                    "route_nodes": path[:5]
                }
            except nx.NetworkXNoPath:
                routes[cluster_key] = {
                    "to_shelter": best_shelter["name"],
                    "distance_km": cdist([cluster_point], [best_shelter_point])[0][0],
                    "estimated_time_hours": "N/A (No route found)",
                    "route_nodes": []
                }

        return {
            "optimized_routes": routes,
            "total_clusters": len(clusters),
            "total_population": sum(c["population"] for c in clusters)
        }

    def assign_shelters_and_transport(self, clusters: list[dict], shelters: list[dict], transport_resources: list[dict]) -> dict:
        """Assign clusters to shelters and transport resources"""
        assignments = {}
        for cluster in clusters:
            cluster_key = f"{cluster['lat']},{cluster['lon']}"
            best_shelter = min(shelters, key=lambda s: cdist([(cluster["lat"], cluster["lon"])], [(s["location"][0], s["location"][1])])[0][0])
            transport = transport_resources[0] if transport_resources else {"type": "walking/foot", "capacity": 1_000_000}
            assignments[cluster_key] = {
                "cluster_population": cluster["population"],
                "shelter_assigned": best_shelter["name"],
                "transport": transport["type"],
                "capacity": transport["capacity"]
            }
        return assignments

    def send_email_notification(self, recipient: dict, notice: dict) -> bool:
        """Simulate sending email notification"""
        logger.info(f"Simulated email sent to {recipient['name']} ({recipient['contact']}): {notice['agenda'][0]}")
        self.email_statuses[recipient["contact"]] = {
            "recipient": recipient["name"],
            "status": "Sent",
            "timestamp": datetime.now().isoformat(),
            "agenda": notice["agenda"][0]
        }
        return True

    def generate_meeting_notices(self, area_name: str, urgency: str = "high") -> list[dict]:
        """Generate meeting notices and simulate email sending"""
        stakeholders = {
            "high": [
                {"name": "Rescue 1122 District Command", "role": "Primary Responder", "contact": "rescue1122@example.com"},
                {"name": "District PDMA Focal Person", "role": "Coordination", "contact": "pdma_district@example.com"},
                {"name": "Local Counselor/Union Council Chairman", "role": "Community Liaison", "contact": "counselor@example.com"},
                {"name": "Red Crescent Local Chapter", "role": "Humanitarian Relief", "contact": "redcrescent@example.com"},
                {"name": "Chief Engineer (Irrigation)", "role": "Technical Support", "contact": "engineer_irrigation@example.com"}
            ],
            "medium": [
                {"name": "Provincial PDMA", "role": "Oversight", "contact": "pdma_province@example.com"},
                {"name": "Rescue 1122 Provincial HQ", "role": "Resource Allocation", "contact": "rescue_hq@example.com"}
            ],
            "low": [
                {"name": "NDMA Observer", "role": "Monitoring", "contact": "ndma@example.com"}
            ]
        }
        notices = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meeting_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        for stakeholder in stakeholders.get(urgency, []):
            notice = {
                "timestamp": now,
                "meeting_time": meeting_time,
                "urgency": urgency.upper(),
                "area": area_name,
                "recipient": stakeholder["name"],
                "role": stakeholder["role"],
                "agenda": [
                    f"Immediate evacuation planning for {area_name}",
                    "Population clusters and vulnerable zones assessment",
                    "Safe evacuation routes and shelter assignments",
                    "Transport resource mobilization (Rescue 1122 + local)",
                    "Coordination with Red Crescent for relief support"
                ],
                "action_items": [
                    "Confirm availability and attendance",
                    "Provide on-ground situation report",
                    "Coordinate with local volunteers/community reps",
                    "Prepare transport and shelter readiness"
                ],
                "contact": stakeholder["contact"],
                "ai_generated": True
            }
            self.send_email_notification(stakeholder, notice)
            notices.append(notice)
        return notices

    def generate_action_plan(self, assignments: dict, routes: dict) -> dict:
        """Generate action plan"""
        total_clusters = len(assignments)
        total_population = sum(assignment["cluster_population"] for assignment in assignments.values())
        plan = {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": total_clusters,
            "total_population": total_population,
            "phases": [
                {
                    "phase": "Immediate (0-2 hours)",
                    "actions": [
                        "Alert all district stakeholders (Rescue 1122, PDMA, Counselors)",
                        "Activate emergency meeting (AI-generated notices sent)",
                        "Mobilize transport resources (buses, trucks, boats)",
                        "Prepare shelters and relief camps"
                    ],
                    "responsible": "Rescue 1122 District Command + Local Counselors"
                },
                {
                    "phase": "Evacuation (2-6 hours)",
                    "actions": [
                        f"Execute evacuation for {total_clusters} clusters ({total_population} people)",
                        "Follow optimized routes to assigned shelters",
                        "Monitor traffic and reroute if blockages occur",
                        "Coordinate with Red Crescent for on-route support"
                    ],
                    "responsible": "Rescue 1122 Teams + Local Volunteers"
                },
                {
                    "phase": "Shelter Management (6-24 hours)",
                    "actions": [
                        "Register evacuees at shelters",
                        "Distribute relief supplies (food, water, medical kits)",
                        "Set up medical camps and counseling services",
                        "Update NDMA/PDMA with occupancy status"
                    ],
                    "responsible": "Red Crescent + Shelter Management Teams"
                },
                {
                    "phase": "Monitoring & Recovery (24+ hours)",
                    "actions": [
                        "Continuous monitoring of flood situation",
                        "Plan phased return when safe",
                        "Damage assessment and rehabilitation planning",
                        "Lessons learned report generation"
                    ],
                    "responsible": "PDMA + NDMA Coordination"
                }
            ],
            "resource_requirements": {
                "transport_vehicles": len([a for a in assignments.values() if a["transport"] != "walking/foot"]),
                "relief_personnel": total_population // 100 + 10,
                "medical_teams": total_clusters // 2 + 1,
                "shelter_capacity_needed": total_population
            },
            "coordination_contacts": [
                {"agency": "Rescue 1122", "contact": "1122", "role": "Primary Response"},
                {"agency": "Red Crescent", "contact": "redcrescent@example.com", "role": "Relief Distribution"},
                {"agency": "Local Counselor", "contact": "counselor@example.com", "role": "Community Liaison"}
            ]
        }
        return plan

    def create_gis_map(self, input_data: EvacuationInput, routes: dict, assignments: dict, output_filename: str = "static/evacuation_map.html"):
        """Create interactive GIS map"""
        m = folium.Map(location=[(input_data.bbox[0] + input_data.bbox[2])/2, (input_data.bbox[1] + input_data.bbox[3])/2], 
                       zoom_start=12)
        if input_data.flood_zones:
            for zone in input_data.flood_zones:
                folium.Circle(
                    location=[zone["lat"], zone["lon"]],
                    radius=zone.get("radius_km", 1) * 1000,
                    popup=f"Flood Zone: {zone['severity']}",
                    color="red",
                    fill=True,
                    fillOpacity=0.3
                ).add_to(m)
        clusters = self.identify_population_clusters(input_data)
        for cluster in clusters:
            folium.Marker(
                [cluster["lat"], cluster["lon"]],
                popup=f"Cluster: {cluster['population']} people, {cluster['vulnerability']} vulnerability",
                icon=folium.Icon(color="orange", icon="users")
            ).add_to(m)
        shelters = input_data.shelters or [
            {"name": "Primary Shelter A", "location": [(input_data.bbox[0] + input_data.bbox[2])/2 + 0.01, (input_data.bbox[1] + input_data.bbox[3])/2 + 0.01], "capacity": 5000},
            {"name": "Secondary Shelter B", "location": [(input_data.bbox[0] + input_data.bbox[2])/2 - 0.01, (input_data.bbox[1] + input_data.bbox[3])/2 - 0.01], "capacity": 3000}
        ]
        for shelter in shelters:
            folium.Marker(
                shelter["location"],
                popup=f"{shelter['name']} - Capacity: {shelter['capacity']}",
                icon=folium.Icon(color="green", icon="home")
            ).add_to(m)
        for cluster_key, route in routes.items():
            if route["route_nodes"]:
                route_coords = [(node[0], node[1]) for node in route["route_nodes"]]
                folium.PolyLine(
                    locations=route_coords,
                    color="blue",
                    weight=3,
                    popup=f"Route to {route['to_shelter']} ({route['distance_km']:.1f} km)"
                ).add_to(m)
        os.makedirs('static', exist_ok=True)
        m.save(output_filename)
        return output_filename

    async def run(self, input_data: EvacuationInput) -> EvacuationOutput:
        try:
            is_valid = await self.validate_input(input_data)
            if not is_valid:
                raise ValueError("Invalid evacuation input")

            clusters = self.identify_population_clusters(input_data)
            gis_data = self.fetch_gis_data(input_data.bbox)

            shelters = input_data.shelters or [
                {"name": "Primary Shelter A", "location": [(input_data.bbox[0] + input_data.bbox[2]) / 2 + 0.01,
                                                        (input_data.bbox[1] + input_data.bbox[3]) / 2 + 0.01],
                "capacity": 5000},
                {"name": "Secondary Shelter B", "location": [(input_data.bbox[0] + input_data.bbox[2]) / 2 - 0.01,
                                                            (input_data.bbox[1] + input_data.bbox[3]) / 2 - 0.01],
                "capacity": 3000}
            ]

            routes = self.optimize_evacuation_routes(clusters, shelters, gis_data)
            assignments = self.assign_shelters_and_transport(clusters, shelters, input_data.transport_resources or [])

            urgency = "high" if any(c["vulnerability"] == "high" for c in clusters) else "medium"
            meeting_notices = self.generate_meeting_notices(input_data.area_name, urgency)
            action_plan = self.generate_action_plan(assignments, routes)
            map_file = self.create_gis_map(input_data, routes["optimized_routes"], assignments)

            return EvacuationOutput(
                evacuation_plan={"clusters": clusters, "total_population": sum(c["population"] for c in clusters), "area_name": input_data.area_name},
                route_optimization=routes,
                shelter_assignment=assignments,
                meeting_notices=meeting_notices,
                action_plan=action_plan,
                visualization_files=[map_file],
                gis_map=map_file
            )
        except Exception as e:
            logger.error(f"EvacuationTool error: {e}")
            raise ValueError(f"EvacuationTool error: {str(e)}")

# -------------------------------
# Agent Handler
# -------------------------------
evac_tool = EvacuationTool()

async def handle_input(data: dict) -> dict:
    """Main handler with guardrails"""
    try:
        validated = EvacuationInput(**data)
    except ValidationError as e:
        return {"status": "error", "message": str(e)}

    result = await evac_tool.run(validated)
    return json.loads(json.dumps(result.model_dump(), cls=CustomJSONEncoder))

# Tool wrapper function
@function_tool
async def evacuation_tool(location: str, population: int, resources: list[str]) -> dict:
    """
    Generate an evacuation plan, optimize routes, assign shelters,
    and coordinate stakeholders for flood-prone areas.
    """
    validated = EvacuationInput(location=location, population=population, resources=resources)
    result = await evac_tool.run(validated)
    return result.model_dump()

evacuation_agent: Agent = Agent(
    name="EvacuationAgent",
    instructions="You are an evacuation planning agent for flood emergencies in Pakistan.",
    model=model,
    tools=[evacuation_tool]   # 👈 Ab yeh tool registered ho gaya
)

# # -------------------------------
# # Example standalone test
# # -------------------------------
# if __name__ == "__main__":
#     sample_input = {
#         "area_name": "Lahore District",
#         "bbox": [31.4, 74.2, 31.7, 74.5],
#         "population_clusters": [
#             {"lat": 31.45, "lon": 74.25, "population": 2500, "vulnerability": "high"},
#             {"lat": 31.55, "lon": 74.35, "population": 1800, "vulnerability": "medium"}
#         ],
#         "flood_zones": [
#             {"lat": 31.48, "lon": 74.28, "radius_km": 2, "severity": "high"}
#         ]
#     }

#     res = asyncio.run(handle_input(sample_input))
#     print(json.dumps(res, indent=2))


# 3️⃣ Async wrapper to run agent
async def run_evacuation_agent(input_data: dict) -> dict:
    """
    input_data example:
    {
        "area_name": "Lahore District",
        "bbox": [31.4, 74.2, 31.7, 74.5],
        "population_clusters": [...],
        "flood_zones": [...],
        "shelters": [...],
        "transport_resources": [...]
    }
    """
    from agents import Runner
    result = await Runner.run(evacuation_agent, [{"role": "user", "content": json.dumps(input_data)}])
    return result.final_output