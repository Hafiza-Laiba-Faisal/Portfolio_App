# agentic_orchestrator.py
"""
Agentic Orchestrator using Google Gemini (genai) as a centralized decision-maker.
Place this file alongside:
 - data_agent.py
 - prediction_agent.py
 - coordination_agent.py
 - evacuation_agent.py  (or the Flask app file)

Run:
  pip install fastapi uvicorn python-dotenv pydantic requests google-genai
  (adjust package name for genai if needed: some installs use `google-genai`)

  Create .env with:
    GEMINI_API_KEY=your_key_here

  Start:
    uvicorn agentic_orchestrator:app --reload --port 8000
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# try import google genai client like your data_agent uses
try:
    from google import genai
except Exception as e:
    genai = None

# try import agents (they should be in same directory)
try:
    import data_agent
except Exception:
    data_agent = None

try:
    import prediction_agent
except Exception:
    prediction_agent = None

try:
    import coordination_agent
except Exception:
    coordination_agent = None

try:
    import evacuation_agent as evac_agent
except Exception:
    evac_agent = None

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LAST_REPORT_PATH = REPORTS_DIR / "last_report.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agentic_orchestrator")

app = FastAPI(title="Agentic Flood Orchestrator (Gemini-driven)", version="0.1")


# ----------------------------
# Pydantic request model
# ----------------------------
class AgentRunRequest(BaseModel):
    goal: str
    preset_area: Optional[str] = None  # e.g., "karachi"
    bbox: Optional[list] = None        # [min_lat, min_lon, max_lat, max_lon]
    max_steps: Optional[int] = 5


# ----------------------------
# Gemini client init
# ----------------------------
client = None
if genai:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set; agentic mode will be unavailable until you set it.")
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini client initialized.")
        except Exception as e:
            logger.exception("Failed to initialize Gemini client: %s", e)
            client = None
else:
    logger.warning("google.genai package not installed or import failed; install google-genai or adjust import.")


# ----------------------------
# Helper I/O functions
# ----------------------------
def save_object(obj: dict, prefix: str):
    ts = int(time.time())
    name = prefix + "_" + str(ts) + ".json"
    path = REPORTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    # update last_report.json
    LAST_REPORT_PATH.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_last_report() -> Optional[dict]:
    if LAST_REPORT_PATH.exists():
        try:
            return json.loads(LAST_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ----------------------------
# Tool wrappers (adapters)
# ----------------------------
# Each tool returns a JSON-serializable dict and should be resilient to missing modules.

def tool_collect(preset_area: Optional[str] = None, bbox: Optional[list] = None, area_name: Optional[str] = None) -> dict:
    if data_agent is None:
        return {"error": "data_agent module not importable on server"}
    if preset_area:
        preset = preset_area.lower()
        if hasattr(data_agent, "PAKISTAN_AREAS") and preset in data_agent.PAKISTAN_AREAS:
            bbox_local = data_agent.PAKISTAN_AREAS[preset]["bbox"]
            name = data_agent.PAKISTAN_AREAS[preset]["name"]
            report = data_agent.collect_flood_data(bbox_local, name)
        else:
            return {"error": f"Unknown preset_area: {preset_area}"}
    elif bbox:
        name = area_name or "custom_area"
        report = data_agent.collect_flood_data(bbox, name)
    else:
        return {"error": "Either preset_area or bbox required"}
    path = save_object(report, "flood_report")
    return {"status": "collected", "report_meta": report.get("meta", {}), "file": path}


def tool_predict(report: Optional[dict] = None) -> dict:
    if prediction_agent is None:
        return {"error": "prediction_agent not importable"}
    if report is None:
        report = load_last_report()
        if report is None:
            return {"error": "No report available to predict on"}
    pred = prediction_agent.predict_flood_risk(report)
    wrapper = {"meta": {"generated_at": int(time.time())}, "prediction": pred, "source_meta": report.get("meta", {})}
    path = save_object(wrapper, "prediction")
    return {"status": "predicted", "prediction": pred, "file": path}


def tool_coordinate(prediction: Optional[dict] = None, area_name: Optional[str] = None) -> dict:
    if coordination_agent is None:
        return {"error": "coordination_agent not importable"}
    if prediction is None:
        # try to load last prediction
        last = load_last_report()
        prediction = last.get("prediction") if last else None
        if prediction is None:
            return {"error": "No prediction provided or found"}
    coord = coordination_agent.FloodMeetingCoordinator()
    meeting = coord.schedule_meeting(prediction, area_name or prediction.get("meta", {}).get("area_name", "Pakistan"))
    coord.send_meeting_notices(meeting)
    meeting_obj = json.loads(json.dumps(meeting.__dict__, default=str))
    path = save_object({"meeting": meeting_obj}, "meeting")
    return {"status": "meeting_scheduled", "meeting": meeting_obj, "file": path}


def tool_evacuate(start_lat: float, start_lon: float) -> dict:
    # If evac_agent exposes a function use it; else, delegate to its HTTP endpoint
    if evac_agent and hasattr(evac_agent, "find_optimal_evacuation_route"):
        best_camp, route = evac_agent.find_optimal_evacuation_route(start_lat, start_lon)
        return {"status": "route", "destination": best_camp, "route": route}
    else:
        # if evac_agent is a Flask app on localhost:5000, call its endpoint
        import requests
        try:
            resp = requests.get(f"http://127.0.0.1:5000/emergency_evacuation?lat={start_lat}&lon={start_lon}", timeout=15)
            return resp.json()
        except Exception as e:
            return {"error": "evacuation service unreachable", "details": str(e)}


# Map of tool names the model can call
TOOLS = {
    "collect": {
        "description": "Collect flood-related data. Args: preset_area (str) OR bbox (list of four floats), area_name (optional). Returns saved report metadata.",
        "func": tool_collect
    },
    "predict": {
        "description": "Run prediction on the latest report or provided report. Args: report (dict, optional).",
        "func": tool_predict
    },
    "coordinate": {
        "description": "Schedule meetings & send notices based on prediction. Args: prediction (dict, optional), area_name (optional).",
        "func": tool_coordinate
    },
    "evacuate": {
        "description": "Generate evacuation route. Args: start_lat (float), start_lon (float).",
        "func": tool_evacuate
    },
    "finish": {
        "description": "Finish workflow and return final summary.",
        "func": lambda **kwargs: {"status": "finished", "summary": kwargs.get("summary", "No summary provided")}
    }
}


# ----------------------------
# Model interaction & planning
# ----------------------------
def build_system_prompt() -> str:
    """
    Provide the model a clear instruction set: available tools, expected JSON action schema,
    and safe-guards.
    """
    tool_texts = []
    for name, info in TOOLS.items():
        tool_texts.append(f"- {name}: {info['description']}")
    tools_block = "\n".join(tool_texts)

    prompt = f"""
You are an autonomous Flood Response Orchestrator assistant. Use the available tools to achieve the user's GOAL.
Available tools:
{tools_block}

When you choose an action, reply with JSON only, in the following schema EXACTLY (no additional text):
{{
  "action": "<tool_name_from_list_or_finish>",
  "args": {{ ... }}   # arguments for the tool; must be JSON serializable
}}

Examples:
{{"action":"collect", "args":{{"preset_area":"karachi"}}}}
{{"action":"predict", "args":{{}}}}
{{"action":"evacuate", "args":{{"start_lat":35.22,"start_lon":72.42}}}}
{{"action":"finish", "args":{{"summary":"All good"}}}}

Rules:
1. Do not output anything except a single JSON object that matches the schema.
2. If information is missing to take a safe action, prefer to call 'collect' to gather data first.
3. Limit the total number of tool calls (the orchestrator will enforce max_steps).
4. After each tool result, you will receive the tool output. Use it to decide the next action.
5. If critical emergency is detected (HIGH risk), prioritize 'coordinate' then 'evacuate' as appropriate.
    """
    return prompt


def call_gemini_once(system_prompt: str, user_goal: str, history: list) -> Optional[dict]:
    """
    Send a single generation request: system_prompt + conversation history + user goal.
    Expect model to return JSON object as described in build_system_prompt.
    """
    if client is None:
        logger.error("Gemini client not initialized.")
        return None

    # Compose contents - small chain of context and recent tool outputs
    contents = [
        {"type": "system", "text": system_prompt},
        {"type": "user", "text": f"GOAL: {user_goal}. Conversation history (most recent last):\n{json.dumps(history[-5:], indent=2) if history else '[]'}"},
        {"type": "assistant", "text": "Respond with a single JSON object indicating the next action."}
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            max_output_tokens=512
        )
        text = response.text.strip()
        # try to extract JSON object (robust)
        try:
            # if model includes surrounding backticks, strip them
            if text.startswith("```"):
                # find first { and last }
                start = text.find("{")
                end = text.rfind("}")
                json_text = text[start:end+1]
            else:
                json_text = text
            obj = json.loads(json_text)
            return obj
        except Exception as e:
            logger.error("Failed to parse model JSON response: %s", e)
            logger.debug("Raw model text: %s", text)
            return None
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        return None


# ----------------------------
# Main agentic loop
# ----------------------------
def run_agentic_loop(goal: str, preset_area: Optional[str], bbox: Optional[list], max_steps: int = 5) -> dict:
    system_prompt = build_system_prompt()
    history = []
    artifacts = {"tool_calls": [], "final_summary": None}
    step = 0

    # Pre-seed context: if preset_area given, seed a collect suggestion
    if preset_area:
        history.append({"note": f"Preset area requested: {preset_area}"})
    if bbox:
        history.append({"note": f"BBox requested: {bbox}"})

    while step < max_steps:
        step += 1
        logger.info("Agentic loop step %d / %d", step, max_steps)

        # Ask model for next action
        model_decision = call_gemini_once(system_prompt, goal, history)
        if model_decision is None:
            # fallback: if no model, stop and return current state
            logger.error("No decision from model. Stopping.")
            artifacts["final_summary"] = {"error": "no_model_decision"}
            break

        action = model_decision.get("action")
        args = model_decision.get("args", {})

        logger.info("Model decided action=%s args=%s", action, args)

        if action not in TOOLS:
            # invalid action: stop
            logger.error("Model requested unknown action: %s", action)
            artifacts["final_summary"] = {"error": f"unknown_action_{action}"}
            break

        # Safety: if action requires bbox/preset and none provided, augment args if we have them
        if action == "collect":
            if "preset_area" not in args and preset_area:
                args["preset_area"] = preset_area
            if "bbox" not in args and bbox:
                args["bbox"] = bbox

        # Execute tool
        try:
            result = TOOLS[action]["func"](**args)
        except Exception as e:
            logger.exception("Tool %s failed: %s", action, e)
            result = {"error": str(e)}

        # record
        artifacts["tool_calls"].append({"step": step, "action": action, "args": args, "result": result})
        history.append({"tool_call": {"action": action, "args": args, "result": result}})

        # if model asked to finish, or tool was 'finish', break
        if action == "finish":
            artifacts["final_summary"] = result.get("summary") if isinstance(result, dict) else str(result)
            break

        # post-tool logic: if predict returns HIGH risk, auto-call coordinate (fast-track)
        try:
            if action == "predict":
                pred = result.get("prediction") or result.get("prediction", result)
                # Try to detect high risk
                if isinstance(pred, dict):
                    level = pred.get("risk_level") or pred.get("prediction", {}).get("risk_level") or pred.get("prediction", {}).get("risk_level")
                    prob = pred.get("probability_score") or pred.get("prediction", {}).get("probability_score")
                    if (level and str(level).upper() == "HIGH") or (prob and float(prob) >= 0.7):
                        # schedule meeting automatically (append a follow-up request in history so model can confirm)
                        history.append({"note": "Auto-scheduling recommended: detected HIGH risk. Coordinator should be invoked."})
                # continue loop and let model choose next
        except Exception:
            pass

        # small guard: if tool returned error, break
        if result and isinstance(result, dict) and result.get("error"):
            logger.warning("Tool returned error: %s", result.get("error"))
            # let model decide next or stop; continue to next iteration

    # After loop, create a human-readable summary of tool calls
    summary = {
        "goal": goal,
        "steps_executed": len(artifacts["tool_calls"]),
        "tool_calls": artifacts["tool_calls"]
    }
    artifacts["final_summary"] = artifacts.get("final_summary") or summary
    save_object(artifacts, "agentic_run")
    return artifacts


# ----------------------------
# FastAPI endpoint
# ----------------------------
@app.post("/api/agent_run")
async def api_agent_run(req: AgentRunRequest, background: BackgroundTasks = None):
    if client is None:
        raise HTTPException(status_code=500, detail="Gemini client not initialized. Set GEMINI_API_KEY in .env and install google-genai.")
    goal = req.goal
    if not goal or goal.strip() == "":
        raise HTTPException(status_code=400, detail="Goal must be a non-empty string.")

    # Run synchronous or background
    if background:
        background.add_task(run_agentic_loop, goal, req.preset_area, req.bbox, req.max_steps or 5)
        return {"status": "started", "message": "Agentic run started in background. Check reports dir for output."}
    else:
        result = run_agentic_loop(goal, req.preset_area, req.bbox, req.max_steps or 5)
        return {"status": "complete", "result": result}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini": bool(client),
        "agents": {
            "data_agent": bool(data_agent),
            "prediction_agent": bool(prediction_agent),
            "coordination_agent": bool(coordination_agent),
            "evacuation_agent": bool(evac_agent)
        }
    }
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic Flood Orchestrator (Gemini-driven)", version="0.1")

# ✅ Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend origin specify kar sakti ho e.g. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# CLI quick-run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agentic_orchestrator:app", host="0.0.0.0", port=8000, reload=True)
