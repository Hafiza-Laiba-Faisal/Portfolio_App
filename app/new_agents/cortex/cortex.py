
# import os
# import json
# import logging
# import asyncio
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from pydantic import BaseModel
# from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, handoff, RunContextWrapper
# from agents.extensions import handoff_filters
# from agents import function_tool
# from app.new_agents.hydro_met.hydro_met import hydro_met_agent
# # ---- Logging ----
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---- Gemini / LLM Setup ----
# gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )
# llm_model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",
#     openai_client=external_client
# )

# # ---- Guardrail Agent ----
# class GuardrailReason(BaseModel):
#     approve: bool
#     reason: str

# llm_guard_agent = Agent(
#     name="LLMGuard",
#     instructions="You are a disaster policy validator. Validate events and decide if approval is needed. Be cautious with mass evacuations and shortages.",
#     model=llm_model,
#     output_type=GuardrailReason
# )

# # ---- Dummy Rule-based Guard ----
# def dummy_guard(payload):
#     if "evacs" in payload and payload["evacs"] > 1000:
#         return True, "Mass evacuation triggered", "high"
#     return False, "No issue", "normal"

# GUARD_MAP = {
#     "hydro_met": (dict, dummy_guard),
#     "evacuation": (dict, dummy_guard),
#     "relief": (dict, dummy_guard),
#     "reconstruction": (dict, dummy_guard),
# }

# # ---- Hybrid Guard (Updated to use asyncio for async compatibility) ----
# async def hybrid_guardrail_async(source: str, payload: dict):
#     model_cls, guard_fn = GUARD_MAP[source]
#     needs_approval, reason, priority = guard_fn(payload)
#     try:
#         res = await Runner.run(llm_guard_agent, [{
#             "role": "user",
#             "content": f"Event from {source}: {json.dumps(payload)}. Rule-based says: {reason}. Approve automatically?"
#         }])
#         llm_output = res.final_output
#         if llm_output.approve is False:
#             needs_approval = True
#             reason = f"LLM override: {llm_output.reason}"
#     except Exception as e:
#         logger.warning(f"LLM guard failed: {e}")
#     return needs_approval, reason, priority

# def hybrid_guardrail(source: str, payload: dict):
#     # Synchronous wrapper for backward compatibility
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         return loop.run_until_complete(hybrid_guardrail_async(source, payload))
#     finally:
#         loop.close()

# # ---- Import Domain Agents ----
# from app.new_agents.hydro_met.hydro_met import hydro_met_agent
# from app.new_agents.evacuation_rout_agents.evacuation_route_agent import evacuation_agent
# from app.new_agents.relief_source import relief_agent
# from app.new_agents.reconstruction.reconstruction import reconstruction_agent

# # ---- Handoff Data ----
# class HandoffData(BaseModel):
#     summary: str
#     query: str

# # ---- Handoff Callback (Fixed to access correct attribute) ----
# def log_the_handoff(ctx: RunContextWrapper[None], input_data: HandoffData):
#     logger.info(f"Context attributes: {dir(ctx)}")  # Debug: see available attributes
#     # ... rest of the function
#     # Fixed: Use ctx.current_agent.name instead of ctx.agent.name
#     try:
#         current_agent = getattr(ctx, 'current_agent', None) or getattr(ctx, 'agent', None)
#         agent_name = current_agent.name if current_agent else "Unknown Agent"
#     except AttributeError:
#         agent_name = "Unknown Agent"
    
#     logger.info(f"Handoff initiated to {agent_name}. Briefing: '{input_data.summary}' for query: '{input_data.query}'")

# # ---- Triage Agent Handoffs ----
# to_hydro_met_handoff = handoff(
#     agent=hydro_met_agent,
#     on_handoff=log_the_handoff,
#     input_type=HandoffData,
#     input_filter=handoff_filters.remove_all_tools
# )

# to_evacuation_handoff = handoff(
#     agent=evacuation_agent,
#     on_handoff=log_the_handoff,
#     input_type=HandoffData,
#     input_filter=handoff_filters.remove_all_tools
# )

# to_relief_handoff = handoff(
#     agent=relief_agent,
#     on_handoff=log_the_handoff,
#     input_type=HandoffData,
#     input_filter=handoff_filters.remove_all_tools
# )

# to_reconstruction_handoff = handoff(
#     agent=reconstruction_agent,
#     on_handoff=log_the_handoff,
#     input_type=HandoffData,
#     input_filter=handoff_filters.remove_all_tools
# )

# # ---- Triage Agent Diagnose Tool ----
# @function_tool
# def diagnose_query(query: str) -> dict:
#     """Diagnose the user query to identify the issue and select the appropriate agent."""
#     query_lower = query.lower()

#     # Hydro-meteorological keywords
#     if "rain" in query_lower or "flood" in query_lower:
#         return {
#             "issue": "hydro_met",
#             "summary": f"Hydro-meteorological issue detected: {query}",
#             "choice": None,
#             "recommended_handoff": "transfer_to_hydro_met_agent"
#         }

#     # City-specific mapping (for hydro-met issues)
#     area_mapping = {
#         "karachi": 1,
#         "lahore": 2,
#         "islamabad": 3,
#         "faisalabad": 4
#     }
    
#     for city, idx in area_mapping.items():
#         if city in query_lower:
#             return {
#                 "issue": "hydro_met",
#                 "summary": f"Hydro-meteorological issue detected: {query}",
#                 "choice": str(idx),
#                 "recommended_handoff": "transfer_to_hydro_met_agent"
#             }

#     # Evacuation-related keywords
#     if "evacuation" in query_lower or "route" in query_lower:
#         return {
#             "issue": "evacuation",
#             "summary": f"Evacuation-related issue detected: {query}",
#             "recommended_handoff": "transfer_to_evacuation_agent"
#         }

#     # Relief-related keywords
#     if "relief" in query_lower or "camp" in query_lower or "food" in query_lower:
#         return {
#             "issue": "relief",
#             "summary": f"Relief-related issue detected: {query}",
#             "recommended_handoff": "transfer_to_relief_agent"
#         }

#     # Reconstruction-related keywords
#     if "reconstruction" in query_lower or "return" in query_lower or "rebuild" in query_lower:
#         return {
#             "issue": "reconstruction",
#             "summary": f"Reconstruction-related issue detected: {query}",
#             "recommended_handoff": "transfer_to_reconstruction_agent"
#         }

#     # If nothing matches
#     return {
#         "issue": "unknown",
#         "summary": f"No suitable agent found for query: {query}",
#         "recommended_handoff": None
#     }

# # ---- Triage Agent ----
# triage_agent = Agent(
#     name="Triage Agent",
#     instructions="""Analyze the user query using the 'diagnose' tool to identify the issue. 
# Based on the diagnosis, hand off to the appropriate specialist agent using the available handoff tools.
# Provide a clear summary of the issue when handing off to the specialist.""",
#     model=llm_model,
#     tools=[
#         diagnose_query
#     ],
#     handoffs=[
#         to_hydro_met_handoff,
#         to_evacuation_handoff,
#         to_relief_handoff,
#         to_reconstruction_handoff
#     ]
# )

# # ---- Flask App ----
# app = Flask(__name__)

# # ---- ADD CORS SUPPORT ----
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Adjust based on your frontend URL
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True
#     }
# })

# # ---- Synchronous wrapper for async Runner.run ----
# def run_agent_sync(agent, messages):
#     """Synchronous wrapper for async Runner.run to avoid event loop issues in Flask threads."""
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop.run_until_complete(Runner.run(agent, messages))
#     finally:
#         loop.close()

# # ---- User → Cortex (frontend) ----
# @app.route("/api/agent_run", methods=["POST", "OPTIONS"])
# def agent_run():
#     # Handle OPTIONS preflight request
#     if request.method == "OPTIONS":
#         return "", 200
    
#     try:
#         data = request.get_json()  # Use get_json() for better error handling
#     except Exception as e:
#         logger.error(f"JSON parsing failed: {e}")
#         return jsonify({"error": "Invalid JSON"}), 400
    
#     user_input = data.get("goal", "") if data else ""
#     logger.info(f"Received query: {user_input}")  # Add logging to debug
    
#     if not user_input:
#         return jsonify({"error": "Missing goal in request"}), 400
    
#     try:
#         logger.info(f"Running triage agent with input: {user_input}")
#         # Use async wrapper for Runner.run instead of run_sync
#         res = run_agent_sync(triage_agent, [{"role": "user", "content": user_input}])
#         final_output = res.final_output
        
#         logger.info(f"Agent response type: {type(final_output)}")  # Debug log
#         logger.info(f"Agent response: {final_output}")  # Debug log
        
#         # Handle different output types more robustly
#         if hasattr(final_output, "model_dump"):
#             try:
#                 final_output = final_output.model_dump()
#                 logger.info(f"After model_dump: {final_output}")
#             except Exception as dump_error:
#                 logger.warning(f"model_dump failed: {dump_error}, using str()")
#                 final_output = str(final_output)
#         elif not isinstance(final_output, (dict, list, str, int, float, bool, type(None))):
#             final_output = str(final_output)
        
#         # Get agent name safely
#         try:
#             agent_name = res.last_agent.name if hasattr(res, "last_agent") and res.last_agent else "Triage Agent"
#         except AttributeError:
#             agent_name = "Triage Agent"
            
#         logger.info(f"Final agent: {agent_name}")  # Debug log
        
#         response_data = {
#             "reply": final_output,
#             "agent": agent_name,
#             "handoff_occurred": agent_name != "Triage Agent"
#         }
        
#         response = jsonify(response_data)
#         logger.info(f"Sending response: {response.get_data(as_text=True)}")  # Debug log
#         return response
        
#     except Exception as e:
#         logger.error(f"Agent run failed: {e}", exc_info=True)  # Include full traceback
#         return jsonify({"error": f"Failed to process query: {str(e)}"}), 500




# # ---- Agents → Cortex (backend events) ----
# @app.route("/publish", methods=["POST", "OPTIONS"])
# def publish_event():
#     # Handle OPTIONS preflight request
#     if request.method == "OPTIONS":
#         return "", 200
    
#     try:
#         data = request.get_json()
#     except Exception as e:
#         logger.error(f"JSON parsing failed: {e}")
#         return jsonify({"error": "Invalid JSON"}), 400
    
#     source = data.get("source") if data else None
#     payload = data.get("payload", {}) if data else {}
    
#     if not source or not payload:
#         return jsonify({"error": "Missing source or payload"}), 400
    
#     try:
#         needs_approval, reason, priority = hybrid_guardrail(source, payload)
#         if needs_approval:
#             return jsonify({
#                 "status": "pending_approval",
#                 "reason": reason,
#                 "priority": priority
#             })
#         return jsonify({
#             "status": "approved",
#             "event": {"source": source, "payload": payload},
#             "priority": priority
#         })
#     except Exception as e:
#         logger.error(f"Event publishing failed: {e}", exc_info=True)
#         return jsonify({"error": f"Failed to process event: {str(e)}"}), 500

# if __name__ == "__main__":
#     # Disable OpenAI tracing warnings since we're using Gemini
#     os.environ["OPENAI_API_KEY"] = ""  # This suppresses the warning
#     app.run(port=8000, debug=True, host='0.0.0.0')  # Allow external connections

import os
import json
import sqlite3
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import folium
import networkx as nx
from scipy.spatial.distance import cdist

from flask import Flask, request, jsonify, render_template_string
from pydantic import BaseModel, field_validator
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

from data_collection import collect_flood_data, PAKISTAN_AREAS
from prediction import predict_flood_risk

# ---------------------------
# 1️⃣ Environment & Model
# ---------------------------
load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

gemini_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)

# ---------------------------
# 2️⃣ HydroMet Models
# ---------------------------
class HydroMetToolInput(BaseModel):
    choice: str
    bbox: list[float] | None = None
    area_name: str | None = None

    @field_validator("bbox")
    def check_bbox(cls, v):
        if v and len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 numbers: min_lat,min_lon,max_lat,max_lon")
        if v:
            min_lat, min_lon, max_lat, max_lon = v
            if not (0 <= min_lat <= 90 and 0 <= max_lat <= 90 and 0 <= min_lon <= 180 and 0 <= max_lon <= 180):
                raise ValueError("Coordinates out of range: lat 0-90, lon 0-180")
        return v

class HydroMetToolOutput(BaseModel):
    prediction_result: dict
    chart_config: dict
    alert_message: str
    visualization_files: list[str]

# ---------------------------
# 3️⃣ HydroMet Tool
# ---------------------------
@function_tool
async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
    # --- Determine area ---
    if input_data.choice == "custom":
        area_name = input_data.area_name or "Custom Area"
        bbox = input_data.bbox or [0, 0, 1, 1]
    else:
        if input_data.choice not in PAKISTAN_AREAS:
            predefined_keys = list(PAKISTAN_AREAS.keys())
            idx = int(input_data.choice) - 1
            area_key = predefined_keys[idx]
            area_config = PAKISTAN_AREAS[area_key]
            bbox = area_config["bbox"]
            area_name = area_config["name"]
        else:
            area_config = PAKISTAN_AREAS[input_data.choice]
            bbox = area_config["bbox"]
            area_name = area_config["name"]

    # --- Collect real flood data & prediction ---
    flood_data = collect_flood_data(bbox, area_name)
    prediction_result = predict_flood_risk(flood_data)

    # --- Generate visualizations ---
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Precipitation chart
    precip = flood_data["data_sources"]["weather"].get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [0]*7)
    plt.figure(figsize=(8,4))
    plt.plot(range(1,len(precip)+1), precip, marker='o', color='b')
    plt.title(f"7-Day Precipitation Forecast for {area_name}")
    plt.xlabel("Day"); plt.ylabel("Precipitation (mm)"); plt.grid(True)
    plt.tight_layout()
    precip_filename = f"{output_dir}/precip_{area_name.replace(' ','_')}_{timestamp}.png"
    plt.savefig(precip_filename); plt.close()

    # Key metrics chart
    metrics = ['Runoff','Peak Discharge','Time to Peak','AMC','Anomaly Score']
    values = [
        prediction_result['numbers']['estimated_runoff_mm'],
        prediction_result['numbers']['estimated_peak_discharge_m3s'],
        prediction_result['numbers']['estimated_time_to_peak_hours'],
        prediction_result['diagnostics']['amc_info']['amc'],
        float(prediction_result['key_factors'][2].split(':')[1])
    ]
    plt.figure(figsize=(8,4))
    bars = plt.bar(metrics, values, color=['blue','green','orange','purple','red'])
    plt.title(f"Key Metrics for {area_name}"); plt.ylabel("Value"); plt.grid(axis='y')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    metrics_filename = f"{output_dir}/metrics_{area_name.replace(' ','_')}_{timestamp}.png"
    plt.savefig(metrics_filename); plt.close()

    visualization_files = [precip_filename, metrics_filename]

    # Alert message
    alert_message = (
        f"Flood Risk Alert for {area_name}:\n"
        f"Risk Level: {prediction_result['risk_level']}\n"
        f"Probability Score: {prediction_result['probability_score']:.2f}\n"
        f"Urdu Summary: {prediction_result['urdu_summary']}\n"
        f"Recommendations: {', '.join(prediction_result['recommendations'])}"
    )

    # Chart config for frontend
    chart_config = {
        "type":"line",
        "data":{"labels":list(range(1,len(precip)+1)),"datasets":[{"label":"Daily Precipitation","data":precip}]}
    }

    # Store in SQLite
    conn = sqlite3.connect('flood_data.db'); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Predictions (
        id INTEGER PRIMARY KEY, area_name TEXT, risk_level TEXT, probability_score REAL,
        timestamp TEXT, visualization_files TEXT
    )''')
    c.execute(
        "INSERT INTO Predictions (area_name,risk_level,probability_score,timestamp,visualization_files) VALUES (?,?,?,?,?)",
        (area_name, prediction_result["risk_level"], prediction_result["probability_score"], timestamp, json.dumps(visualization_files))
    )
    conn.commit(); conn.close()

    return HydroMetToolOutput(
        prediction_result=prediction_result,
        chart_config=chart_config,
        alert_message=alert_message,
        visualization_files=visualization_files
    ).model_dump()

# ---------------------------
# 4️⃣ HydroMet Agent
# ---------------------------
hydro_met_agent = Agent(
    name="HydroMetAgent",
    instructions="Use HydroMet tool for flood predictions in Pakistan",
    model=model,
    tools=[hydro_met_tool_fn]
)

# ---------------------------
# 5️⃣ Evacuation Models
# ---------------------------
class EvacuationInput(BaseModel):
    area_name: str
    bbox: list[float]
    population_clusters: list[dict] | None = None
    flood_zones: list[dict] | None = None
    transport_resources: list[dict] | None = None
    shelters: list[dict] | None = None

    @field_validator("bbox")
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 coordinates")
        min_lat, min_lon, max_lat, max_lon = v
        if min_lat >= max_lat or min_lon >= max_lon:
            raise ValueError("min coordinates must be smaller than max")
        return v

class EvacuationOutput(BaseModel):
    evacuation_plan: dict
    route_optimization: dict
    shelter_assignment: dict
    meeting_notices: list[dict]
    action_plan: dict
    visualization_files: list[str]
    gis_map: str

# ---------------------------
# 6️⃣ Evacuation Tool
# ---------------------------
class EvacuationTool:
    def __init__(self):
        self.email_statuses = {}

    async def run(self, input_data: EvacuationInput) -> EvacuationOutput:
        clusters = self.identify_population_clusters(input_data)
        gis_data = self.fetch_gis_data(input_data.bbox)
        shelters = input_data.shelters or self.default_shelters(input_data.bbox)
        routes = self.optimize_evacuation_routes(clusters, shelters, gis_data)
        assignments = self.assign_shelters_and_transport(clusters, shelters, input_data.transport_resources or [])
        urgency = "high" if any(c["vulnerability"]=="high" for c in clusters) else "medium"
        meeting_notices = self.generate_meeting_notices(input_data.area_name, urgency)
        action_plan = self.generate_action_plan(assignments, routes)
        map_file = self.create_gis_map(input_data, routes["optimized_routes"], assignments)
        return EvacuationOutput(
            evacuation_plan={"clusters": clusters, "total_population": sum(c["population"] for c in clusters),
                             "area_name": input_data.area_name},
            route_optimization=routes,
            shelter_assignment=assignments,
            meeting_notices=meeting_notices,
            action_plan=action_plan,
            visualization_files=[map_file],
            gis_map=map_file
        )

    # --- All methods (identify_population_clusters, fetch_gis_data, optimize_evacuation_routes, etc.) remain same as your original script ---
    # Add them here exactly as in your EvacuationTool class

evac_tool = EvacuationTool()

async def run_evacuation_agent(input_data: dict) -> dict:
    validated = EvacuationInput(**input_data)
    result = await evac_tool.run(validated)
    return json.loads(json.dumps(result.model_dump()))

evacuation_agent = Agent(
    name="EvacuationAgent",
    instructions="Flood evacuation planning for Pakistan districts",
    model=model,
    tools=[]
)

# ---------------------------
# 7️⃣ Flask App
# ---------------------------
app = Flask(__name__, static_folder="visualizations")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Flood Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>body{padding:20px} iframe{width:100%;height:500px;border:1px solid #ccc;} .section{margin-bottom:40px;}</style>
</head>
<body>
<h1>Flood Prediction & Evacuation Dashboard</h1>

<div class="section"><h3>HydroMet Precipitation</h3><img id="precip" src="" width="800"/></div>
<div class="section"><h3>Key Metrics</h3><img id="metrics" src="" width="800"/></div>
<div class="section"><h3>Evacuation GIS Map</h3><iframe src="/static/evac_map.html"></iframe></div>
<div class="section"><h3>Routes Table</h3><table class="table table-striped" id="routes-table"><thead><tr><th>Cluster</th><th>Shelter</th><th>Distance</th><th>Time (h)</th></tr></thead><tbody></tbody></table></div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script>
async function loadData(){
  // HydroMet
  const hydro_resp = await fetch('/flood_forecast',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({"choice":"1"})});
  const hydro_data = await hydro_resp.json();
  $("#precip").attr("src", hydro_data.visualization_files[0]);
  $("#metrics").attr("src", hydro_data.visualization_files[1]);

  // Evacuation
  const evac_resp = await fetch('/evacuation_demo',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({"area_name":"Lahore District","bbox":[31.4,74.2,31.7,74.5]})});
  const evac_data = await evac_resp.json();
  const tbody = $("#routes-table tbody"); tbody.empty();
  const routes = evac_data.route_optimization.optimized_routes;
  for(const [cluster, route] of Object.entries(routes)){
    tbody.append(`<tr><td>${cluster}</td><td>${route.to_shelter}</td><td>${route.distance_km}</td><td>${route.estimated_time_hours}</td></tr>`);
  }
}
$(document).ready(()=>loadData());
</script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/flood_forecast", methods=["POST"])
async def flood_forecast():
    data = await request.get_json(force=True)
    choice = data.get("choice","1")
    bbox = data.get("bbox", None)
    area_name = data.get("area_name", None)
    result = await Runner.run(hydro_met_agent, [{"role":"user","content":json.dumps({
        "choice": choice, "bbox": bbox, "area_name": area_name
    })}])
    return jsonify(result.final_output)
@app.route("/api/agent_run", methods=["POST"])
async def agent_run():
    data = await request.get_json(force=True)
    agent_type = data.get("agent_type")  # "hydro_met" or "evacuation"
    payload = data.get("input_data", {})

    if agent_type == "hydro_met":
        result = await Runner.run(hydro_met_agent, [{"role":"user","content":json.dumps(payload)}])
        return jsonify(result.final_output)
    elif agent_type == "evacuation":
        result = await run_evacuation_agent(payload)
        return jsonify(result)
    else:
        return jsonify({"error": "Invalid agent_type"}), 400


@app.route("/evacuation_demo", methods=["POST"])
async def evacuation_demo():
    data = await request.get_json(force=True)
    result = await run_evacuation_agent(data)
    return jsonify(result)

# ---------------------------
# 8️⃣ Run App
# ---------------------------
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    app.run(host="0.0.0.0", port=8000, debug=True)
