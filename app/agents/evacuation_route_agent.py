import os
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
import requests
import folium
from folium import plugins
import networkx as nx
from scipy.spatial.distance import cdist
from flask import Flask, render_template, jsonify
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# Configure OpenAI client for Gemini
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Validation Models
class FloodInputSanitizer(BaseModel):
    is_valid: bool
    reason: str | None = None

# Evacuation Models
class EvacuationInput(BaseModel):
    area_name: str
    bbox: list[float]  # [min_lat, min_lon, max_lat, max_lon]
    population_clusters: list[dict] | None = None
    flood_zones: list[dict] | None = None
    transport_resources: list[dict] | None = None
    shelters: list[dict] | None = None

class EvacuationOutput(BaseModel):
    evacuation_plan: dict
    route_optimization: dict
    shelter_assignment: dict
    meeting_notices: list[dict]
    action_plan: dict
    visualization_files: list[str]
    gis_map: str

class EvacuationTool:
    def __init__(self):
        self.db_conn = sqlite3.connect('evacuation_data.db')
        self.setup_database()
        self.graph = None
        self.email_statuses = {}  # Track email statuses

    def setup_database(self):
        """Setup SQLite database"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS EvacuationPlans (
            id INTEGER PRIMARY KEY,
            area_name TEXT,
            bbox TEXT,
            timestamp TEXT,
            population_clusters TEXT,
            flood_zones TEXT,
            transport_resources TEXT,
            shelters TEXT,
            evacuation_plan TEXT,
            route_optimization TEXT,
            shelter_assignment TEXT,
            meeting_notices TEXT,
            action_plan TEXT,
            email_statuses TEXT,
            status TEXT
        )
        ''')
        self.db_conn.commit()

    async def validate_input(self, input_data: EvacuationInput) -> bool:
        """Validate evacuation input using Gemini model"""
        try:
            input_str = json.dumps(input_data.dict())
            prompt = (
                f"Validate evacuation input for flood response. Check if area_name is valid, bbox has 4 coordinates within 0-90 lat/0-180 lon, "
                f"population_clusters have valid lat/lon/population/vulnerability, flood_zones have valid lat/lon/radius/severity, "
                f"transport_resources have valid type/capacity/location, shelters have valid name/location/capacity/type. "
                f"Input: {input_str}\nReturn JSON: {{'is_valid': bool, 'reason': str | null}}"
            )
            response = await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"[EVACUATION INPUT VALIDATION] {result}")
            return result.get('is_valid', False)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

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
        routes = {}
        for cluster in clusters:
            cluster_point = (cluster["lat"], cluster["lon"])
            best_shelter = min(shelters, key=lambda s: cdist([cluster_point], [(s["location"][0], s["location"][1])])[0][0])
            best_shelter_point = (best_shelter["location"][0], best_shelter["location"][1])
            try:
                path = nx.shortest_path(G, cluster_point, best_shelter_point, weight="weight")
                routes[(cluster["lat"], cluster["lon"])] = {
                    "to_shelter": best_shelter["name"],
                    "distance_km": cdist([cluster_point], [best_shelter_point])[0][0],
                    "estimated_time_hours": cdist([cluster_point], [best_shelter_point])[0][0] / 50,
                    "route_nodes": path[:3]
                }
            except nx.NetworkXNoPath:
                routes[(cluster["lat"], cluster["lon"])] = {
                    "to_shelter": best_shelter["name"],
                    "distance_km": cdist([cluster_point], [best_shelter_point])[0][0],
                    "estimated_time_hours": "N/A (No route found)",
                    "route_nodes": []
                }
        return {"optimized_routes": routes, "total_clusters": len(clusters), "total_population": sum(c["population"] for c in clusters)}

    def assign_shelters_and_transport(self, clusters: list[dict], shelters: list[dict], transport: list[dict]) -> dict:
        """Assign shelters and transport resources"""
        total_population = sum(cluster["population"] for cluster in clusters)
        shelters_sorted = sorted(shelters, key=lambda s: s["capacity"], reverse=True)
        clusters_sorted = sorted(clusters, key=lambda c: c.get("evac_priority", 0), reverse=True)
        assignments = {}
        remaining_capacity = {s["name"]: s["capacity"] for s in shelters}
        remaining_transport = {t["type"]: t["capacity"] for t in transport or []}
        for cluster in clusters_sorted:
            assigned_shelter = None
            for shelter_name, capacity in remaining_capacity.items():
                if capacity >= cluster["population"]:
                    assigned_shelter = shelter_name
                    remaining_capacity[shelter_name] -= cluster["population"]
                    break
            if not assigned_shelter:
                assigned_shelter = min(remaining_capacity, key=remaining_capacity.get)
                remaining_capacity[assigned_shelter] -= cluster["population"]
            assigned_transport = None
            for t_type, t_capacity in remaining_transport.items():
                if t_capacity >= cluster["population"]:
                    assigned_transport = t_type
                    remaining_transport[t_type] -= cluster["population"]
                    break
            assignments[str(cluster["lat"]) + "," + str(cluster["lon"])] = {
                "cluster_population": cluster["population"],
                "vulnerability": cluster["vulnerability"],
                "shelter": assigned_shelter,
                "transport": assigned_transport or "walking/foot",
                "priority": cluster.get("evac_priority", 0)
            }
        return {
            "shelter_assignments": assignments,
            "total_population": total_population,
            "shelter_utilization": {k: max(0, v) for k, v in remaining_capacity.items()},
            "transport_utilization": {k: max(0, v) for k, v in remaining_transport.items()}
        }

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
        """Main evacuation workflow"""
        try:
            is_valid = await self.validate_input(input_data)
            if not is_valid:
                raise ValueError("Invalid evacuation input")
            clusters = self.identify_population_clusters(input_data)
            gis_data = self.fetch_gis_data(input_data.bbox)
            routes = self.optimize_evacuation_routes(clusters, input_data.shelters or [], gis_data)
            assignments = self.assign_shelters_and_transport(clusters, input_data.shelters or [], input_data.transport_resources or [])
            urgency = "high" if any(c["vulnerability"] == "high" for c in clusters) else "medium"
            meeting_notices = self.generate_meeting_notices(input_data.area_name, urgency)
            action_plan = self.generate_action_plan(assignments, routes)
            map_file = self.create_gis_map(input_data, routes["optimized_routes"], assignments)
            cursor = self.db_conn.cursor()
            cursor.execute('''
            INSERT INTO EvacuationPlans (area_name, bbox, timestamp, population_clusters, flood_zones, transport_resources, shelters,
            evacuation_plan, route_optimization, shelter_assignment, meeting_notices, action_plan, email_statuses, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                input_data.area_name,
                json.dumps(input_data.bbox),
                datetime.now().isoformat(),
                json.dumps(clusters),
                json.dumps(input_data.flood_zones or []),
                json.dumps(input_data.transport_resources or []),
                json.dumps(input_data.shelters or []),
                json.dumps({"total_clusters": len(clusters), "vulnerable_clusters": len([c for c in clusters if c["vulnerability"] == "high"])}),
                json.dumps(routes),
                json.dumps(assignments),
                json.dumps(meeting_notices),
                json.dumps(action_plan),
                json.dumps(self.email_statuses),
                "completed"
            ))
            self.db_conn.commit()
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

# Flask Routes
@app.route('/')
def dashboard():
    """Render main evacuation dashboard"""
    cursor = app.evac_tool.db_conn.cursor()
    cursor.execute("SELECT * FROM EvacuationPlans WHERE status = 'completed' ORDER BY timestamp DESC LIMIT 1")
    plan = cursor.fetchone()
    if not plan:
        return render_template('dashboard.html', plan=None, email_statuses={})
    
    plan_data = {
        "area_name": plan[1],
        "timestamp": plan[3],
        "evacuation_plan": json.loads(plan[7]),
        "route_optimization": json.loads(plan[8]),
        "shelter_assignment": json.loads(plan[9]),
        "meeting_notices": json.loads(plan[10]),
        "action_plan": json.loads(plan[11]),
        "email_statuses": json.loads(plan[12]),
        "gis_map": "static/evacuation_map.html"
    }
    return render_template('dashboard.html', plan=plan_data, email_statuses=plan_data["email_statuses"])

@app.route('/run_evacuation', methods=['POST'])
async def run_evacuation():
    """Run evacuation planning via API"""
    input_data = EvacuationInput(
        area_name="Lahore District",
        bbox=[31.4, 74.2, 31.7, 74.5],
        population_clusters=[
            {"lat": 31.45, "lon": 74.25, "population": 2500, "vulnerability": "high"},
            {"lat": 31.55, "lon": 74.35, "population": 1800, "vulnerability": "medium"},
            {"lat": 31.65, "lon": 74.45, "population": 3200, "vulnerability": "low"}
        ],
        flood_zones=[
            {"lat": 31.48, "lon": 74.28, "radius_km": 2, "severity": "high"},
            {"lat": 31.58, "lon": 74.38, "radius_km": 1.5, "severity": "medium"}
        ],
        transport_resources=[
            {"type": "bus", "capacity": 50, "location": [31.50, 74.30]},
            {"type": "truck", "capacity": 30, "location": [31.52, 74.32]},
            {"type": "boat", "capacity": 20, "location": [31.46, 74.26]}
        ],
        shelters=[
            {"name": "Government College Shelter", "location": [31.60, 74.40], "capacity": 5000, "type": "primary"},
            {"name": "Lahore Fort Camp", "location": [31.58, 74.35], "capacity": 3000, "type": "secondary"},
            {"name": "Model Town School", "location": [31.48, 74.30], "capacity": 2000, "type": "secondary"}
        ]
    )
    try:
        result = await app.evac_tool.run(input_data)
        return jsonify({
            "status": "success",
            "evacuation_plan": result.evacuation_plan,
            "route_optimization": result.route_optimization,
            "shelter_assignment": result.shelter_assignment,
            "meeting_notices": result.meeting_notices,
            "action_plan": result.action_plan,
            "gis_map": result.gis_map,
            "email_statuses": app.evac_tool.email_statuses
        })
    except Exception as e:
        logger.error(f"Evacuation API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# HTML Template for Dashboard
@app.route('/template')
def create_template():
    """Create dashboard.html template"""
    template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flood Evacuation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        #map { height: 500px; width: 100%; }
        .card { background-color: #f9fafb; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
</head>
<body class="bg-gray-100 font-sans">
    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold text-center mb-6">Flood Evacuation Dashboard</h1>
        {% if plan %}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Evacuation Summary -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-2">Evacuation Summary</h2>
                <p><strong>Area:</strong> {{ plan.area_name }}</p>
                <p><strong>Timestamp:</strong> {{ plan.timestamp }}</p>
                <p><strong>Total Population:</strong> {{ plan.evacuation_plan.total_population | int }} people</p>
                <p><strong>Vulnerable Clusters:</strong> {{ plan.evacuation_plan.vulnerable_clusters }}</p>
            </div>
            <!-- Route Optimization -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-2">Route Optimization</h2>
                <p><strong>Total Clusters:</strong> {{ plan.route_optimization.total_clusters }}</p>
                <p><strong>Average Distance:</strong> {{ plan.route_optimization.optimized_routes | map(attribute='distance_km') | select('float') | list | average | round(1) }} km</p>
                <p><strong>Estimated Total Time:</strong> {{ plan.route_optimization.optimized_routes | map(attribute='estimated_time_hours') | select('float') | list | sum | round(1) }} hours</p>
            </div>
            <!-- Shelter Assignment -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-2">Shelter Assignment</h2>
                <p><strong>Total Shelters Used:</strong> {{ plan.shelter_assignment.shelter_assignments | map(attribute='shelter') | unique | list | length }}</p>
                <p><strong>Transport Vehicles:</strong> {{ plan.shelter_assignment.shelter_assignments | selectattr('transport', 'ne', 'walking/foot') | list | length }}</p>
                <table class="w-full mt-2 border">
                    <tr class="bg-gray-200">
                        <th class="p-2">Cluster</th>
                        <th class="p-2">Population</th>
                        <th class="p-2">Shelter</th>
                        <th class="p-2">Transport</th>
                    </tr>
                    {% for key, assignment in plan.shelter_assignment.shelter_assignments.items() %}
                    <tr>
                        <td class="p-2 border">{{ key }}</td>
                        <td class="p-2 border">{{ assignment.cluster_population }}</td>
                        <td class="p-2 border">{{ assignment.shelter }}</td>
                        <td class="p-2 border">{{ assignment.transport }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            <!-- Email Notifications -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-2">Email Notifications</h2>
                <table class="w-full border">
                    <tr class="bg-gray-200">
                        <th class="p-2">Recipient</th>
                        <th class="p-2">Status</th>
                        <th class="p-2">Timestamp</th>
                        <th class="p-2">Agenda</th>
                    </tr>
                    {% for contact, status in email_statuses.items() %}
                    <tr>
                        <td class="p-2 border">{{ status.recipient }}</td>
                        <td class="p-2 border">{{ status.status }}</td>
                        <td class="p-2 border">{{ status.timestamp }}</td>
                        <td class="p-2 border">{{ status.agenda }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            <!-- GIS Map -->
            <div class="card col-span-2">
                <h2 class="text-xl font-semibold mb-2">Evacuation Routes Map</h2>
                <iframe src="{{ plan.gis_map }}" style="width: 100%; height: 500px; border: none;"></iframe>
            </div>
        </div>
        {% else %}
        <p class="text-center text-gray-600">No evacuation plans available. Run the evacuation tool to generate a plan.</p>
        {% endif %}
        <div class="text-center mt-4">
            <form action="/run_evacuation" method="POST">
                <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                    Run Evacuation Planning
                </button>
            </form>
        </div>
    </div>
</body>
</html>
    """
    os.makedirs('templates', exist_ok=True)
    with open('templates/dashboard.html', 'w') as f:
        f.write(template_content)
    return "Template created", 200

# Initialize EvacuationTool
app.evac_tool = EvacuationTool()

# Main Execution
if __name__ == "__main__":
    # Create template if it doesn't exist
    app.test_client().get('/template')
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)