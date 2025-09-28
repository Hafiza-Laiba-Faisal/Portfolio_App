# # import os
# # import json
# # import logging
# # from datetime import datetime, timedelta
# # from typing import Dict, Any, List
# # from dotenv import load_dotenv

# # # Google SDK Imports
# # import google.generativeai as genai
# # from google.cloud import aiplatform
# # from google.api_core import client_options
# # from google.cloud.aiplatform.gapic.schema import predict

# # # LangGraph for Agentic Flow (pip install langgraph)
# # from langgraph.graph import StateGraph, END
# # from langgraph.prebuilt import ToolNode, tools_condition

# # # Your Existing Agents (import from your files)
# # from data_agent import collect_flood_data, PAKISTAN_AREAS  # Data collection
# # from prediction_agent import predict_flood_risk  # Deterministic prediction

# # load_dotenv()
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "your-project-id")  # Set in .env
# # LOCATION = "us-central1"

# # # Configure Google SDK
# # genai.configure(api_key=GEMINI_API_KEY)
# # aiplatform.init(project=PROJECT_ID, location=LOCATION)

# # # Gemini LLM Setup
# # llm = genai.GenerativeModel('gemini-2.0-flash-exp')

# # # Shared State for Agent Synchronization
# # class AgentState:
# #     def __init__(self):
# #         self.area_name: str = ""
# #         self.bbox: List[float] = []
# #         self.flood_data: Dict[str, Any] = {}
# #         self.prediction: Dict[str, Any] = {}
# #         self.meeting_triggers: List[str] = []  # e.g., ["Early Warning", "Emergency"]
# #         self.evactuation_plan: Dict[str, Any] = {}
# #         self.messages: List[str] = []  # Chat history for Gemini
# #         self.risk_level: str = "LOW"  # From prediction
# #         self.timestamp: str = datetime.now().isoformat()

# # # Tool 1: Data Collection Agent
# # def data_collection_tool(state: AgentState, area_key: str) -> Dict[str, Any]:
# #     """Collect flood data using data_agent.py."""
# #     logging.info(f"Data Agent: Collecting data for {area_key}")
# #     if area_key not in PAKISTAN_AREAS:
# #         return {"error": f"Area {area_key} not found."}
# #     area_config = PAKISTAN_AREAS[area_key]
# #     bbox = area_config["bbox"]
# #     name = area_config["name"]
# #     flood_data = collect_flood_data(bbox, name)
# #     state.flood_data = flood_data
# #     state.area_name = name
# #     state.bbox = bbox
# #     state.messages.append(f"Data collected for {name}: Risk factors - {flood_data.get('risk_assessment', {}).get('overall_risk', 'LOW')}")
# #     return state.__dict__

# # # Tool 2: Prediction Agent
# # def prediction_tool(state: AgentState) -> Dict[str, Any]:
# #     """Run deterministic prediction using prediction_agent.py."""
# #     logging.info("Prediction Agent: Running flood risk prediction")
# #     if not state.flood_data:
# #         return {"error": "No flood data available."}
# #     prediction = predict_flood_risk(state.flood_data)
# #     state.prediction = prediction
# #     state.risk_level = prediction.get("risk_level", "LOW")
# #     state.messages.append(f"Prediction complete: {state.risk_level} (Score: {prediction.get('probability_score', 0):.2f})")
# #     return state.__dict__

# # # Tool 3: Coordination Agent (Meeting Orchestration)
# # def coordination_tool(state: AgentState) -> Dict[str, Any]:
# #     """Orchestrate meetings based on risk level using Gemini."""
# #     logging.info("Coordination Agent: Orchestrating meetings")
# #     risk = state.risk_level
# #     prompt = f"""
# #     Based on flood risk {risk} for {state.area_name}:
# #     - If LOW: Schedule weekly monitoring meeting (Federal Technical: PMD, FFC, IRSA, PCIW, PCRWR + NDMA).
# #     - If MEDIUM (>50%): Trigger Early Warning meeting (Federal + Provincial PDMAs/Irrigation + NDMA/Chief Secretaries).
# #     - If HIGH (>70%): Emergency Response meeting (National Command: PM/NDMA/CMs + Provincial Ops: PDMA, Rescue 1122 + District Teams).
# #     - Recovery: Weekly rehab meeting (NDMA, PDMAs, NHA, Rescue 1122, Local Counselors).
    
# #     Generate:
# #     - Meeting type and participants.
# #     - Agenda (tailored: summaries for leaders, simulations for engineers, evacuation for locals).
# #     - Auto-notice template (email/WhatsApp).
# #     - Frequency (e.g., daily for emergency).
# #     """
# #     response = llm.generate_content(prompt)
# #     meeting_plan = {
# #         "type": "Emergency" if risk == "HIGH" else "Early Warning" if risk == "MEDIUM" else "Monitoring",
# #         "participants": "NDMA, PDMA, Rescue 1122, Local Counselors" if risk == "HIGH" else "PMD, FFC, IRSA",
# #         "agenda": response.text[:500],  # Truncate for brevity
# #         "notice_template": f"Urgent Flood Meeting: Risk {risk} in {state.area_name}. Join via Google Meet: [link].",
# #         "frequency": "Daily" if risk == "HIGH" else "Weekly"
# #     }
# #     state.meeting_triggers.append(json.dumps(meeting_plan))
# #     state.messages.append(f"Meeting scheduled: {meeting_plan['type']} for {risk} risk.")
# #     return state.__dict__

# # # Tool 4: Evacuation Agent
# # def evacuation_tool(state: AgentState) -> Dict[str, Any]:
# #     """Generate evacuation plan using Gemini and Google Maps API."""
# #     logging.info("Evacuation Agent: Generating plan")
# #     if state.risk_level not in ["MEDIUM", "HIGH"]:
# #         return {"note": "No evacuation needed for LOW risk."}
# #     prompt = f"""
# #     For {state.area_name} (bbox: {state.bbox}), risk {state.risk_level}:
# #     1. Map vulnerable areas (low-lying, near rivers like Kabul River).
# #     2. Allocate safe zones (schools, higher ground) using NADRA-like data.
# #     3. Optimize routes with Google Maps (distance, capacity).
# #     4. Mobilize resources (Rescue 1122 vehicles, Red Crescent supplies).
# #     5. Communication: SMS to locals, calls to counselors.
# #     6. Monitoring: Track evacuees via dashboard.
    
# #     Output JSON: vulnerable_areas, safe_zones, routes, resources, alerts.
# #     """
# #     response = llm.generate_content(prompt)
# #     # Parse response as JSON (assume Gemini returns structured)
# #     try:
# #         evac_plan = json.loads(response.text)  # Or use Gemini's structured output
# #     except:
# #         evac_plan = {"vulnerable_areas": "Low-lying villages near Kabul River", "safe_zones": "Schools in higher ground", "routes": "Optimized via Google Maps", "resources": "1122 vehicles + tents", "alerts": "SMS: Evacuate to [zone] immediately."}
# #     state.evactuation_plan = evac_plan
# #     state.messages.append(f"Evacuation plan generated for {len(evac_plan.get('vulnerable_areas', []))} areas.")
# #     return state.__dict__

# # # Gemini Supervisor Tool
# # def supervisor_tool(state: AgentState) -> Dict[str, Any]:
# #     """Gemini LLM as Supervisor: Decide next action based on state."""
# #     logging.info("Gemini Supervisor: Deciding next action")
# #     prompt = f"""
# #     Current State for {state.area_name}:
# #     - Risk Level: {state.risk_level}
# #     - Data Collected: {'Yes' if state.flood_data else 'No'}
# #     - Prediction Done: {'Yes' if state.prediction else 'No'}
# #     - Meetings Triggered: {len(state.meeting_triggers)}
# #     - Evacuation Needed: {'Yes' if state.risk_level in ['MEDIUM', 'HIGH'] else 'No'}

# #     Rules:
# #     - If no data: Run data collection.
# #     - If data but no prediction: Run prediction.
# #     - If prediction done: Run coordination for meetings.
# #     - If risk MEDIUM/HIGH: Run evacuation.
# #     - Always: Generate final report.

# #     Respond with JSON: {{"next_action": "collect_data" | "predict" | "coordinate" | "evacuate" | "finalize", "reason": "brief reason"}}
# #     """
# #     response = llm.generate_content(prompt)
# #     try:
# #         decision = json.loads(response.text)
# #         next_action = decision.get("next_action", "finalize")
# #         state.messages.append(f"Supervisor decision: {next_action} - {decision.get('reason', '')}")
# #         return {"next_action": next_action}
# #     except:
# #         return {"next_action": "finalize"}

# # # Build Agentic Graph (Synchronization Flow)
# # def build_agentic_graph():
# #     workflow = StateGraph(AgentState)
    
# #     # Add Nodes (Agents)
# #     workflow.add_node("data_agent", lambda state: data_collection_tool(state, state.get("input_area", "peshawar")))
# #     workflow.add_node("prediction_agent", lambda state: prediction_tool(state))
# #     workflow.add_node("coordination_agent", lambda state: coordination_tool(state))
# #     workflow.add_node("evacuation_agent", lambda state: evacuation_tool(state))
# #     workflow.add_node("supervisor", lambda state: supervisor_tool(state))
    
# #     # Conditional Edges (Based on Supervisor Decision)
# #     def route_to_action(state: AgentState):
# #         action = state.get("next_action", "finalize")
# #         if action == "collect_data":
# #             return "data_agent"
# #         elif action == "predict":
# #             return "prediction_agent"
# #         elif action == "coordinate":
# #             return "coordination_agent"
# #         elif action == "evacuate":
# #             return "evacuation_agent"
# #         else:
# #             return END
    
# #     workflow.add_conditional_edges("supervisor", route_to_action)
# #     workflow.set_entry_point("supervisor")
    
# #     # Compile Graph
# #     app = workflow.compile()
# #     return app

# # # Main Execution
# # def run_flood_agentic_system(area_key: str):
# #     """Run the full agentic system for given area."""
# #     logging.info(f"Starting Agentic Flood System for {area_key}")
# #     app = build_agentic_graph()
    
# #     initial_state = AgentState()
# #     initial_state.input_area = area_key  # Input for data agent
    
# #     # Invoke Graph (Synchronized Flow)
# #     result = app.invoke(initial_state.__dict__)
    
# #     # Generate Final Report with Gemini
# #     final_prompt = f"""
# #     Generate Final Flood Coordination Report for {result['area_name']}:
# #     - Risk: {result['risk_level']}
# #     - Prediction: {json.dumps(result['prediction'], indent=1)[:500]}
# #     - Meetings: {result['meeting_triggers']}
# #     - Evacuation: {json.dumps(result['evactuation_plan'], indent=1)[:500]}
    
# #     Structure as JSON: {{"executive_summary": "1-page dashboard text", "stakeholder_actions": {{"national": [...], "provincial": [...], "district": [...]}}, "urdu_advisory": "2 lines in Urdu"}}
# #     """
# #     final_response = llm.generate_content(final_prompt)
# #     try:
# #         final_report = json.loads(final_response.text)
# #     except:
# #         final_report = {"executive_summary": final_response.text[:300], "stakeholder_actions": {}, "urdu_advisory": "سیلاب کا خطرہ کم ہے، نگرانی جاری رکھیں۔"}
    
# #     # Save Report
# #     report = {
# #         "timestamp": datetime.now().isoformat(),
# #         "area": result['area_name'],
# #         "state": result,
# #         "final_report": final_report
# #     }
# #     filename = f"flood_agentic_report_{area_key}_{int(time.time())}.json"
# #     with open(filename, 'w', encoding='utf-8') as f:
# #         json.dump(report, f, indent=2, ensure_ascii=False)
# #     logging.info(f"Report saved to {filename}")
    
# #     return report

# # if __name__ == "__main__":
# #     area_key = input("Enter area key (e.g., peshawar): ").strip().lower()
# #     report = run_flood_agentic_system(area_key)
# #     print("\n📊 Final Agentic System Report:")
# #     print(json.dumps(report['final_report'], indent=2, ensure_ascii=False))




# """
# meeting_coordination_agent.py

# AI-Based Flood Meeting Coordination Agent for Pakistan
# Automates meetings based on flood risk levels and coordinates stakeholders
# """

# import json
# import smtplib
# import logging
# from datetime import datetime, timedelta
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from dataclasses import dataclass
# from typing import List, Dict, Optional
# from enum import Enum

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class FloodStage(Enum):
#     MONITORING = "monitoring"
#     EARLY_WARNING = "early_warning" 
#     EMERGENCY = "emergency"
#     RECOVERY = "recovery"

# class MeetingType(Enum):
#     ROUTINE = "routine"
#     ALERT = "alert"
#     EMERGENCY = "emergency"
#     RECOVERY = "recovery"

# @dataclass
# class Stakeholder:
#     name: str
#     email: str
#     phone: str
#     organization: str
#     level: str  # national, federal, provincial, district
#     role: str

# @dataclass
# class Meeting:
#     meeting_id: str
#     meeting_type: MeetingType
#     flood_stage: FloodStage
#     datetime_scheduled: datetime
#     duration_hours: int
#     participants: List[Stakeholder]
#     agenda_items: List[str]
#     location: str
#     virtual_link: Optional[str] = None

# class FloodMeetingCoordinator:
#     def __init__(self):
#         self.stakeholders = self._initialize_stakeholders()
#         self.meeting_history = []
#         self.active_meetings = []
        
#     def _initialize_stakeholders(self) -> Dict[str, List[Stakeholder]]:
#         """Initialize Pakistan flood response stakeholders by level"""
#         return {
#             "national": [
#                 Stakeholder("President Secretariat", "president@gov.pk", "+92-51-9204801", "Presidency", "national", "decision_maker"),
#                 Stakeholder("PM Office", "pmo@pmo.gov.pk", "+92-51-9202404", "PMO", "national", "decision_maker"),
#                 Stakeholder("Chairman NDMA", "chairman@ndma.gov.pk", "+92-51-9205600", "NDMA", "national", "coordinator")
#             ],
#             "federal": [
#                 Stakeholder("DG PMD", "dg@pmd.gov.pk", "+92-51-9250368", "PMD", "federal", "technical"),
#                 Stakeholder("Chairman FFC", "chairman@ffc.gov.pk", "+92-51-9205071", "FFC", "federal", "technical"),
#                 Stakeholder("Chairman IRSA", "chairman@irsa.gov.pk", "+92-51-9244820", "IRSA", "federal", "technical"),
#                 Stakeholder("PCIW Commissioner", "pciw@mowr.gov.pk", "+92-51-9202171", "PCIW", "federal", "technical")
#             ],
#             "provincial": [
#                 Stakeholder("DG PDMA Punjab", "dg@pdma.gop.pk", "+92-42-99200645", "PDMA Punjab", "provincial", "operational"),
#                 Stakeholder("DG PDMA Sindh", "dg@pdmasindh.gos.pk", "+92-21-99205849", "PDMA Sindh", "provincial", "operational"),
#                 Stakeholder("DG PDMA KPK", "dg@pdma.gokp.pk", "+92-91-9213151", "PDMA KPK", "provincial", "operational"),
#                 Stakeholder("DG PDMA Balochistan", "dg@pdmabalochistan.gob.pk", "+92-81-9203045", "PDMA Balochistan", "provincial", "operational")
#             ],
#             "district": [
#                 Stakeholder("Rescue 1122 Command", "control@rescue.gov.pk", "1122", "Rescue 1122", "district", "response"),
#                 Stakeholder("Red Crescent District", "district@prcs.org.pk", "+92-51-9250404", "Pakistan Red Crescent", "district", "relief"),
#                 Stakeholder("Local Councillor Rep", "councillor@local.gov.pk", "+92-300-1234567", "Local Government", "district", "community")
#             ]
#         }

#     def determine_flood_stage(self, flood_risk_data: Dict) -> FloodStage:
#         """Determine current flood stage based on risk assessment"""
#         try:
#             risk_level = flood_risk_data.get("risk_level", "LOW")
#             probability = flood_risk_data.get("probability_score", 0.0)
            
#             if risk_level == "HIGH" or probability > 0.7:
#                 return FloodStage.EMERGENCY
#             elif risk_level == "MEDIUM" or probability > 0.5:
#                 return FloodStage.EARLY_WARNING
#             else:
#                 return FloodStage.MONITORING
                
#         except Exception as e:
#             logging.error(f"Error determining flood stage: {e}")
#             return FloodStage.MONITORING

#     def select_participants(self, flood_stage: FloodStage, area_type: str = "general") -> List[Stakeholder]:
#         """Select appropriate participants based on flood stage"""
#         participants = []
        
#         if flood_stage == FloodStage.MONITORING:
#             # Routine monitoring - federal technical only
#             participants.extend(self.stakeholders["federal"])
#             participants.extend([s for s in self.stakeholders["national"] if s.organization == "NDMA"])
            
#         elif flood_stage == FloodStage.EARLY_WARNING:
#             # Add provincial coordination
#             participants.extend(self.stakeholders["federal"])
#             participants.extend(self.stakeholders["provincial"])
#             participants.extend([s for s in self.stakeholders["national"] if s.role in ["coordinator"]])
            
#         elif flood_stage == FloodStage.EMERGENCY:
#             # All levels activated
#             for level in self.stakeholders.values():
#                 participants.extend(level)
                
#         elif flood_stage == FloodStage.RECOVERY:
#             # Recovery coordination
#             participants.extend(self.stakeholders["federal"])
#             participants.extend(self.stakeholders["provincial"])
#             participants.extend([s for s in self.stakeholders["district"] if s.organization in ["Pakistan Red Crescent", "Local Government"]])
            
#         return participants

#     def generate_agenda(self, flood_stage: FloodStage, flood_data: Dict) -> List[str]:
#         """Generate meeting agenda based on flood stage and data"""
#         base_items = [
#             f"Meeting called: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
#             "Attendance confirmation"
#         ]
        
#         if flood_stage == FloodStage.MONITORING:
#             agenda = base_items + [
#                 "Weekly weather outlook (PMD)",
#                 "Dam levels and water releases (IRSA)",
#                 "7-day flood risk assessment",
#                 "Routine maintenance updates",
#                 "Next review date"
#             ]
            
#         elif flood_stage == FloodStage.EARLY_WARNING:
#             agenda = base_items + [
#                 "URGENT: Flood probability >50% detected",
#                 "Current weather situation and 48h forecast",
#                 "Dam release strategies (IRSA/PCIW)",
#                 "Provincial preparedness status",
#                 "Public warning protocols activation",
#                 "Resource pre-positioning",
#                 "Next situation update timing"
#             ]
            
#         elif flood_stage == FloodStage.EMERGENCY:
#             agenda = base_items + [
#                 "EMERGENCY: Active flooding or >70% probability",
#                 "Real-time situation report (SitRep)",
#                 "Rescue operations status (1122)",
#                 "Evacuation status and shelter occupancy",
#                 "Road closures and infrastructure damage",
#                 "Medical emergency response",
#                 "Media and public communication",
#                 "Next emergency briefing (12h)"
#             ]
            
#         elif flood_stage == FloodStage.RECOVERY:
#             agenda = base_items + [
#                 "Post-flood damage assessment",
#                 "Relief distribution status",
#                 "Infrastructure rehabilitation priorities", 
#                 "Return and resettlement planning",
#                 "Financial assistance coordination",
#                 "Lessons learned documentation",
#                 "Recovery timeline review"
#             ]
            
#         # Add specific data points if available
#         if flood_data.get("numbers"):
#             numbers = flood_data["numbers"]
#             agenda.append(f"Technical data: {numbers.get('forecast_precip_next7_mm', 0):.1f}mm forecast, Risk score: {flood_data.get('probability_score', 0):.2f}")
            
#         return agenda

#     def schedule_meeting(self, flood_risk_data: Dict, area_name: str = "Pakistan") -> Meeting:
#         """Schedule appropriate meeting based on flood risk"""
#         try:
#             # Determine stage and meeting type
#             flood_stage = self.determine_flood_stage(flood_risk_data)
            
#             # Set meeting timing based on urgency
#             if flood_stage == FloodStage.EMERGENCY:
#                 meeting_time = datetime.now() + timedelta(hours=2)  # 2 hours notice
#                 meeting_type = MeetingType.EMERGENCY
#                 duration = 1
#             elif flood_stage == FloodStage.EARLY_WARNING:
#                 meeting_time = datetime.now() + timedelta(hours=6)  # 6 hours notice  
#                 meeting_type = MeetingType.ALERT
#                 duration = 2
#             elif flood_stage == FloodStage.RECOVERY:
#                 meeting_time = datetime.now() + timedelta(days=1)  # Next day
#                 meeting_type = MeetingType.RECOVERY
#                 duration = 3
#             else:
#                 meeting_time = datetime.now() + timedelta(days=7)  # Weekly routine
#                 meeting_type = MeetingType.ROUTINE
#                 duration = 1
                
#             # Select participants
#             participants = self.select_participants(flood_stage)
            
#             # Generate agenda
#             agenda = self.generate_agenda(flood_stage, flood_risk_data)
            
#             # Create meeting
#             meeting = Meeting(
#                 meeting_id=f"FLOOD_{flood_stage.value.upper()}_{int(datetime.now().timestamp())}",
#                 meeting_type=meeting_type,
#                 flood_stage=flood_stage,
#                 datetime_scheduled=meeting_time,
#                 duration_hours=duration,
#                 participants=participants,
#                 agenda_items=agenda,
#                 location="NDMA Headquarters / Virtual",
#                 virtual_link="https://meet.gov.pk/flood-response"
#             )
            
#             self.active_meetings.append(meeting)
#             logging.info(f"Meeting scheduled: {meeting.meeting_id} for {meeting_time}")
            
#             return meeting
            
#         except Exception as e:
#             logging.error(f"Error scheduling meeting: {e}")
#             raise

#     def send_meeting_notices(self, meeting: Meeting) -> Dict[str, bool]:
#         """Send meeting notices to all participants"""
#         results = {}
        
#         for participant in meeting.participants:
#             try:
#                 # Generate notice content
#                 notice = self._generate_meeting_notice(meeting, participant)
                
#                 # Send email (mock implementation)
#                 success = self._send_email_notice(participant.email, notice)
#                 results[participant.name] = success
                
#                 # Log WhatsApp/SMS (mock)
#                 if success:
#                     logging.info(f"Notice sent to {participant.name} ({participant.organization})")
                    
#             except Exception as e:
#                 logging.error(f"Failed to send notice to {participant.name}: {e}")
#                 results[participant.name] = False
                
#         return results

#     def _generate_meeting_notice(self, meeting: Meeting, participant: Stakeholder) -> Dict:
#         """Generate personalized meeting notice"""
#         urgency = "HIGH PRIORITY" if meeting.flood_stage in [FloodStage.EMERGENCY, FloodStage.EARLY_WARNING] else "ROUTINE"
        
#         notice = {
#             "subject": f"[{urgency}] Flood Response Meeting - {meeting.flood_stage.value.replace('_', ' ').title()}",
#             "body": f"""
# Assalam-o-Alaikum / Dear {participant.name},

# You are required to attend an urgent flood response coordination meeting:

# MEETING DETAILS:
# Meeting ID: {meeting.meeting_id}
# Type: {meeting.meeting_type.value.title()} ({meeting.flood_stage.value.replace('_', ' ').title()})
# Date & Time: {meeting.datetime_scheduled.strftime('%Y-%m-%d at %H:%M PKT')}
# Duration: {meeting.duration_hours} hour(s)
# Location: {meeting.location}
# Virtual Link: {meeting.virtual_link}

# AGENDA:
# {chr(10).join([f"• {item}" for item in meeting.agenda_items])}

# Your attendance is {"MANDATORY" if meeting.flood_stage == FloodStage.EMERGENCY else "required"}.

# Please confirm your attendance by replying to this notice.

# Best regards,
# AI Flood Coordination System
# National Disaster Management Authority (NDMA)
# Government of Pakistan
#             """,
#             "priority": "high" if meeting.flood_stage in [FloodStage.EMERGENCY, FloodStage.EARLY_WARNING] else "normal"
#         }
        
#         return notice

#     def _send_email_notice(self, email: str, notice: Dict) -> bool:
#         """Mock email sending function"""
#         # In real implementation, integrate with SMTP server
#         logging.info(f"EMAIL SENT to {email}: {notice['subject']}")
#         return True

#     def generate_meeting_dashboard(self, meeting: Meeting, live_data: Dict = None) -> Dict:
#         """Generate real-time dashboard for meeting"""
#         dashboard = {
#             "meeting_info": {
#                 "id": meeting.meeting_id,
#                 "stage": meeting.flood_stage.value,
#                 "participants_count": len(meeting.participants),
#                 "start_time": meeting.datetime_scheduled.isoformat()
#             },
#             "flood_status": live_data or {},
#             "action_items": [],
#             "decisions": [],
#             "next_meeting": None
#         }
        
#         if live_data:
#             # Add specific dashboard elements based on stage
#             if meeting.flood_stage == FloodStage.EMERGENCY:
#                 dashboard["emergency_metrics"] = {
#                     "evacuees": live_data.get("evacuees", 0),
#                     "rescue_teams_deployed": live_data.get("rescue_teams", 0),
#                     "roads_blocked": live_data.get("blocked_roads", 0),
#                     "shelters_active": live_data.get("active_shelters", 0)
#                 }
                
#         return dashboard

#     def track_meeting_outcomes(self, meeting: Meeting, outcomes: Dict) -> None:
#         """Track meeting decisions and actions"""
#         meeting_record = {
#             "meeting_id": meeting.meeting_id,
#             "completion_time": datetime.now().isoformat(),
#             "decisions": outcomes.get("decisions", []),
#             "action_items": outcomes.get("action_items", []),
#             "next_review": outcomes.get("next_review_date"),
#             "attendees": [p.name for p in meeting.participants]
#         }
        
#         self.meeting_history.append(meeting_record)
        
#         # Remove from active meetings
#         self.active_meetings = [m for m in self.active_meetings if m.meeting_id != meeting.meeting_id]
        
#         logging.info(f"Meeting {meeting.meeting_id} completed and recorded")

#     def get_meeting_frequency_stats(self) -> Dict:
#         """Get meeting frequency statistics"""
#         total_meetings = len(self.meeting_history)
#         stage_counts = {}
        
#         for record in self.meeting_history:
#             stage = record.get("stage", "unknown")
#             stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
#         return {
#             "total_meetings": total_meetings,
#             "by_stage": stage_counts,
#             "active_meetings": len(self.active_meetings)
#         }

# def main():
#     """Demo function"""
#     coordinator = FloodMeetingCoordinator()
    
#     # Mock flood risk data
#     mock_risk_data = {
#         "risk_level": "HIGH",
#         "probability_score": 0.8,
#         "numbers": {
#             "forecast_precip_next7_mm": 150.0,
#             "estimated_runoff_mm": 45.0
#         }
#     }
    
#     # Schedule meeting
#     meeting = coordinator.schedule_meeting(mock_risk_data, "Karachi")
    
#     # Send notices
#     notice_results = coordinator.send_meeting_notices(meeting)
    
#     # Generate dashboard
#     dashboard = coordinator.generate_meeting_dashboard(meeting, mock_risk_data)
    
#     # Print results
#     print(f"Meeting scheduled: {meeting.meeting_id}")
#     print(f"Participants: {len(meeting.participants)}")
#     print(f"Notices sent: {sum(notice_results.values())}/{len(notice_results)}")
#     print(f"Dashboard generated with {len(dashboard)} sections")

# if __name__ == "__main__":
#     main()



"""
flood_agentic_system_upgraded.py
Upgraded AI-Based Flood Meeting Coordination Agent for Pakistan
- More stakeholders (Governance, Defence, NHA, PCRWR, GHQ engineers)
- Dynamic meeting frequency rules
- Executive / Technical / Community summaries (mock)
- Evacuation planning (basic optimizer)
- SitRep and Lessons Learned generation
- Mock notification system (email log)
"""

import json
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uuid
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Enums & Dataclasses ---
class FloodStage(Enum):
    MONITORING = "monitoring"
    EARLY_WARNING = "early_warning"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"

class MeetingType(Enum):
    ROUTINE = "routine"
    ALERT = "alert"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"

@dataclass
class Stakeholder:
    name: str
    email: str
    phone: str
    organization: str
    level: str  # national, federal, provincial, district
    role: str

@dataclass
class Meeting:
    meeting_id: str
    meeting_type: MeetingType
    flood_stage: FloodStage
    datetime_scheduled: datetime
    duration_hours: int
    participants: List[Stakeholder]
    agenda_items: List[str]
    location: str
    virtual_link: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

# --- Coordinator Class ---
class FloodMeetingCoordinator:
    def __init__(self):
        self.stakeholders = self._initialize_stakeholders_full()
        self.meeting_history: List[Dict] = []
        self.active_meetings: List[Meeting] = []
        # threshold config (can be externalized)
        self.thresholds = {
            "early_warning_prob": 0.5,
            "emergency_prob": 0.7
        }

    def _initialize_stakeholders_full(self) -> Dict[str, List[Stakeholder]]:
        """Extended stakeholder list including governance, defence, NHA, PCRWR, GHQ, etc."""
        base = {
            "national": [
                Stakeholder("President Secretariat", "president@gov.pk", "+92-51-9204801", "Presidency", "national", "decision_maker"),
                Stakeholder("PM Office", "pmo@pmo.gov.pk", "+92-51-9202404", "PMO", "national", "decision_maker"),
                Stakeholder("Chairman NDMA", "chairman@ndma.gov.pk", "+92-51-9205600", "NDMA", "national", "coordinator"),
                Stakeholder("Ministry of Defence", "defence@moD.gov.pk", "+92-51-9200000", "Ministry of Defence", "national", "support"),
                Stakeholder("Finance Ministry", "finance@moF.gov.pk", "+92-51-9201000", "Finance", "national", "finance")
            ],
            "federal": [
                Stakeholder("DG PMD", "dg@pmd.gov.pk", "+92-51-9250368", "PMD", "federal", "technical"),
                Stakeholder("Chairman FFC", "chairman@ffc.gov.pk", "+92-51-9205071", "FFC", "federal", "technical"),
                Stakeholder("Chairman IRSA", "chairman@irsa.gov.pk", "+92-51-9244820", "IRSA", "federal", "technical"),
                Stakeholder("PCIW Commissioner", "pciw@mowr.gov.pk", "+92-51-9202171", "PCIW", "federal", "technical"),
                Stakeholder("PCRWR Lead", "lead@pcrwr.gov.pk", "+92-51-9203000", "PCRWR", "federal", "research")
            ],
            "provincial": [
                Stakeholder("DG PDMA Punjab", "dg@pdma.gop.pk", "+92-42-99200645", "PDMA Punjab", "provincial", "operational"),
                Stakeholder("DG PDMA Sindh", "dg@pdmasindh.gos.pk", "+92-21-99205849", "PDMA Sindh", "provincial", "operational"),
                Stakeholder("DG PDMA KPK", "dg@pdma.gokp.pk", "+92-91-9213151", "PDMA KPK", "provincial", "operational"),
                Stakeholder("Chief Secretary Punjab", "cs@punjab.gov.pk", "+92-42-99200000", "Chief Secretariat", "provincial", "admin")
            ],
            "district": [
                Stakeholder("Rescue 1122 Command", "control@rescue.gov.pk", "1122", "Rescue 1122", "district", "response"),
                Stakeholder("Red Crescent District", "district@prcs.org.pk", "+92-51-9250404", "Pakistan Red Crescent", "district", "relief"),
                Stakeholder("Local Councillor Rep", "councillor@local.gov.pk", "+92-300-1234567", "Local Government", "district", "community")
            ],
            "infra": [
                Stakeholder("NHA Director", "director@nha.gov.pk", "+92-51-9209000", "NHA", "federal", "infrastructure"),
                Stakeholder("GHQ Engineers", "eng@ghq.gov.pk", "+92-51-9209999", "GHQ", "national", "engineering")
            ]
        }
        return base

    # --- Stage Determination ---
    def determine_flood_stage(self, flood_risk_data: Dict) -> FloodStage:
        """Determine current flood stage based on risk assessment"""
        try:
            risk_level = flood_risk_data.get("risk_level", "").upper()
            probability = float(flood_risk_data.get("probability_score", 0.0))
            if risk_level == "HIGH" or probability >= self.thresholds["emergency_prob"]:
                return FloodStage.EMERGENCY
            elif risk_level == "MEDIUM" or probability >= self.thresholds["early_warning_prob"]:
                return FloodStage.EARLY_WARNING
            elif risk_level == "RECOVERY":
                return FloodStage.RECOVERY
            else:
                return FloodStage.MONITORING
        except Exception as e:
            logging.error(f"Error determining flood stage: {e}")
            return FloodStage.MONITORING

    # --- Participant Selection ---
    def select_participants(self, flood_stage: FloodStage, affected_provinces: Optional[List[str]] = None) -> List[Stakeholder]:
        """Select appropriate participants based on flood stage and affected areas"""
        participants = []
        # base picks
        if flood_stage == FloodStage.MONITORING:
            participants.extend(self.stakeholders["federal"])
            participants.extend([s for s in self.stakeholders["national"] if s.organization == "NDMA"])
        elif flood_stage == FloodStage.EARLY_WARNING:
            participants.extend(self.stakeholders["federal"])
            participants.extend(self.stakeholders["provincial"])
            participants.extend([s for s in self.stakeholders["national"] if s.role in ["coordinator"]])
        elif flood_stage == FloodStage.EMERGENCY:
            # include all levels + infra + national decision makers
            for level in self.stakeholders.values():
                participants.extend(level)
        elif flood_stage == FloodStage.RECOVERY:
            participants.extend(self.stakeholders["federal"])
            participants.extend(self.stakeholders["provincial"])
            participants.extend(self.stakeholders["infra"])
            participants.extend([s for s in self.stakeholders["district"] if s.organization in ["Pakistan Red Crescent", "Local Government"]])

        # deduplicate by email
        unique = {}
        for s in participants:
            unique[s.email] = s
        return list(unique.values())

    # --- Agenda Generation ---
    def generate_agenda(self, flood_stage: FloodStage, flood_data: Dict) -> List[str]:
        base_items = [
            f"Meeting called: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "Attendance confirmation"
        ]
        # reuse previous structure but add infra/evacuation points
        if flood_stage == FloodStage.MONITORING:
            agenda = base_items + [
                "Weekly weather outlook (PMD)",
                "Dam levels and water releases (IRSA)",
                "7-day flood risk assessment",
            ]
        elif flood_stage == FloodStage.EARLY_WARNING:
            agenda = base_items + [
                "URGENT: Flood probability exceeded threshold",
                "48h forecast & dam release plans",
                "Provincial preparedness & resource pre-positioning",
                "Evacuation readiness checklists"
            ]
        elif flood_stage == FloodStage.EMERGENCY:
            agenda = base_items + [
                "EMERGENCY SITUATION REPORT (SitRep)",
                "Rescue & Evacuation updates",
                "Infrastructure damage (NHA / GHQ engineers)",
                "Medical & Relief operations (Red Crescent / Hospitals)",
                "Public Communications & Media Strategy"
            ]
        else:
            agenda = base_items + [
                "Post-flood damage assessment",
                "Relief distribution & rehabilitation plans",
                "Financial allocations & donor coordination",
                "Lessons learned / future mitigation planning"
            ]

        if flood_data.get("numbers"):
            numbers = flood_data["numbers"]
            agenda.append(f"Technical data: {numbers.get('forecast_precip_next7_mm', 0):.1f}mm forecast, Risk score: {flood_data.get('probability_score', 0):.2f}")

        return agenda

    # --- Meeting Scheduling with Dynamic Frequency ---
    def schedule_meeting(self, flood_risk_data: Dict, area_name: str = "Pakistan", affected_provinces: Optional[List[str]] = None) -> Meeting:
        """Schedule meeting using dynamic frequency rules (tighter scheduling for high urgency)"""
        flood_stage = self.determine_flood_stage(flood_risk_data)

        # frequency logic (customizable)
        if flood_stage == FloodStage.EMERGENCY:
            meeting_time = datetime.now() + timedelta(hours=2)
            meeting_type = MeetingType.EMERGENCY
            duration = 1
        elif flood_stage == FloodStage.EARLY_WARNING:
            meeting_time = datetime.now() + timedelta(hours=6)
            meeting_type = MeetingType.ALERT
            duration = 2
        elif flood_stage == FloodStage.RECOVERY:
            meeting_time = datetime.now() + timedelta(days=1)
            meeting_type = MeetingType.RECOVERY
            duration = 3
        else:
            meeting_time = datetime.now() + timedelta(days=7)
            meeting_type = MeetingType.ROUTINE
            duration = 1

        participants = self.select_participants(flood_stage, affected_provinces)
        agenda = self.generate_agenda(flood_stage, flood_risk_data)

        meeting = Meeting(
            meeting_id=f"FLOOD_{flood_stage.value.upper()}_{uuid.uuid4().hex[:8]}",
            meeting_type=meeting_type,
            flood_stage=flood_stage,
            datetime_scheduled=meeting_time,
            duration_hours=duration,
            participants=participants,
            agenda_items=agenda,
            location="NDMA Headquarters / Virtual",
            virtual_link="https://meet.gov.pk/flood-response"
        )

        self.active_meetings.append(meeting)
        logging.info(f"Meeting scheduled: {meeting.meeting_id} for {meeting_time.isoformat()}")
        return meeting

    # --- Mock AI Summaries (executive, technical, community) ---
    def generate_executive_summary(self, flood_data: Dict) -> str:
        """Produce a short executive summary (mock)"""
        prob = flood_data.get("probability_score", 0.0)
        precip = flood_data.get("numbers", {}).get("forecast_precip_next7_mm", 0.0)
        summary = f"Executive Summary: Flood probability {prob:.2f}. Expected precipitation {precip:.1f} mm over next 7 days. Recommended actions: Prepare national command, pre-position resources, alert provinces."
        return summary

    def generate_technical_summary(self, flood_data: Dict) -> str:
        """Mock technical summary with more details"""
        precip = flood_data.get("numbers", {}).get("forecast_precip_next7_mm", 0.0)
        inflow = flood_data.get("numbers", {}).get("estimated_runoff_mm", 0.0)
        summary = f"Technical Summary: Forecast precip {precip:.1f} mm; estimated runoff {inflow:.1f} mm. Dam inflows rising; consider graduated releases and monitor IRSA gauges."
        return summary

    def generate_community_summary(self, flood_data: Dict, local_tips: Optional[str] = None) -> str:
        tips = local_tips or "Stay on higher ground; follow local authority instructions; keep emergency kit ready."
        return f"Community Advisory: {tips}"

    # --- Evacuation Planning (basic optimization) ---
    def plan_evacuation(self, area_population_data: List[Dict], safe_sites: List[Dict]) -> Dict:
        """
        Simple greedy allocation:
        - area_population_data: list of {"area":"X","population":int,"priority":float, "distance_matrix":{site_id:km}}
        - safe_sites: list of {"site_id":"s1","capacity":int,"location":"..."}
        Returns allocation plan and summary metrics.
        """
        # Flatten structures and sort areas by priority descending
        areas = sorted(area_population_data, key=lambda x: x.get("priority", 1.0), reverse=True)
        sites = {s["site_id"]: {"capacity": s["capacity"], "location": s.get("location")} for s in safe_sites}
        allocation = {s_id: [] for s_id in sites.keys()}
        unallocated = []

        for area in areas:
            remaining = area["population"]
            # sort sites by distance (closest first) if distance matrix present else by largest capacity
            distances = area.get("distance_matrix")
            site_order = list(sites.keys())
            if distances:
                site_order.sort(key=lambda s: distances.get(s, math.inf))
            else:
                site_order.sort(key=lambda s: -sites[s]["capacity"])

            for s_id in site_order:
                if remaining <= 0:
                    break
                cap = sites[s_id]["capacity"]
                if cap <= 0:
                    continue
                allocate_count = min(cap, remaining)
                allocation[s_id].append({"area": area["area"], "count": allocate_count})
                sites[s_id]["capacity"] -= allocate_count
                remaining -= allocate_count

            if remaining > 0:
                unallocated.append({"area": area["area"], "left": remaining})

        # compute metrics
        total_evacuated = sum(sum(x["count"] for x in lst) for lst in allocation.values())
        total_population = sum(a["population"] for a in area_population_data)
        return {
            "allocation": allocation,
            "unallocated": unallocated,
            "total_evacuated": total_evacuated,
            "total_population": total_population,
            "evacuation_coverage": total_evacuated / total_population if total_population else 0.0
        }

    # --- Notifications (mock) ---
    def _generate_meeting_notice(self, meeting: Meeting, participant: Stakeholder) -> Dict:
        urgency = "HIGH PRIORITY" if meeting.flood_stage in [FloodStage.EMERGENCY, FloodStage.EARLY_WARNING] else "ROUTINE"
        subject = f"[{urgency}] Flood Response Meeting - {meeting.flood_stage.value.replace('_',' ').title()}"
        body = f"""
Assalam-o-Alaikum / Dear {participant.name},

You are requested to attend:

Meeting ID: {meeting.meeting_id}
Type: {meeting.meeting_type.value.title()} ({meeting.flood_stage.value.replace('_',' ').title()})
Date & Time: {meeting.datetime_scheduled.strftime('%Y-%m-%d at %H:%M PKT')}
Duration: {meeting.duration_hours} hour(s)
Location: {meeting.location}
Virtual Link: {meeting.virtual_link}

AGENDA:
{chr(10).join([f"• {item}" for item in meeting.agenda_items])}

Please confirm attendance.

Regards,
AI Flood Coordination System
"""
        return {"subject": subject, "body": body, "priority": "high" if meeting.flood_stage in [FloodStage.EMERGENCY, FloodStage.EARLY_WARNING] else "normal"}

    def _send_email_notice(self, email: str, notice: Dict) -> bool:
        """Mock: log email send. Replace with real SMTP or gateway in production."""
        logging.info(f"EMAIL SENT to {email}: {notice['subject']}")
        # Here you'd construct MIME and send via smtplib if configured.
        return True

    def send_meeting_notices(self, meeting: Meeting) -> Dict[str, bool]:
        results = {}
        for participant in meeting.participants:
            try:
                notice = self._generate_meeting_notice(meeting, participant)
                success = self._send_email_notice(participant.email, notice)
                results[participant.name] = success
                if success:
                    logging.info(f"Notice logged for {participant.name} ({participant.organization})")
            except Exception as e:
                logging.error(f"Failed to send notice to {participant.name}: {e}")
                results[participant.name] = False
        return results

    # --- SitRep & Lessons Learned ---
    def generate_sitrep(self, live_data: Dict) -> Dict:
        """Generate a SitRep summary from live incident feeds (mocked)"""
        sitrep = {
            "timestamp": datetime.now().isoformat(),
            "summary": f"Current evacuees: {live_data.get('evacuees',0)}, active_shelters: {live_data.get('active_shelters',0)}, roads_blocked: {live_data.get('blocked_roads',0)}",
            "detailed": live_data
        }
        return sitrep

    def track_meeting_outcomes(self, meeting: Meeting, outcomes: Dict) -> None:
        """Record meeting outcomes and produce lessons learned stub"""
        meeting_record = {
            "meeting_id": meeting.meeting_id,
            "stage": meeting.flood_stage.value,
            "completion_time": datetime.now().isoformat(),
            "decisions": outcomes.get("decisions", []),
            "action_items": outcomes.get("action_items", []),
            "next_review": outcomes.get("next_review_date"),
            "attendees": [p.name for p in meeting.participants],
            "lessons_learned": self._generate_lessons_learned(outcomes)
        }
        self.meeting_history.append(meeting_record)
        self.active_meetings = [m for m in self.active_meetings if m.meeting_id != meeting.meeting_id]
        logging.info(f"Meeting {meeting.meeting_id} recorded with outcomes.")

    def _generate_lessons_learned(self, outcomes: Dict) -> List[str]:
        """Produce lessons-learned points from outcomes (mock)"""
        lessons = []
        if outcomes.get("delays_reported"):
            lessons.append("Improve early-warning dissemination and local comms.")
        if outcomes.get("resource_gaps"):
            lessons.append("Pre-position resources based on highest-risk districts.")
        if not lessons:
            lessons.append("No immediate lessons captured; continue monitoring.")
        return lessons

    # --- Dashboard Generation ---
    def generate_meeting_dashboard(self, meeting: Meeting, live_data: Dict = None) -> Dict:
        dashboard = {
            "meeting_info": {
                "id": meeting.meeting_id,
                "stage": meeting.flood_stage.value,
                "participants_count": len(meeting.participants),
                "start_time": meeting.datetime_scheduled.isoformat()
            },
            "flood_status": live_data or {},
            "executive_summary": self.generate_executive_summary(live_data or {}),
            "technical_summary": self.generate_technical_summary(live_data or {}),
            "community_summary": self.generate_community_summary(live_data or {})
        }
        if live_data and meeting.flood_stage == FloodStage.EMERGENCY:
            dashboard["emergency_metrics"] = {
                "evacuees": live_data.get("evacuees", 0),
                "rescue_teams_deployed": live_data.get("rescue_teams", 0),
                "roads_blocked": live_data.get("blocked_roads", 0),
                "shelters_active": live_data.get("active_shelters", 0)
            }
        return dashboard

# --- Demo main ---
def main():
    coordinator = FloodMeetingCoordinator()

    mock_risk_data = {
        "risk_level": "HIGH",
        "probability_score": 0.82,
        "numbers": {
            "forecast_precip_next7_mm": 180.0,
            "estimated_runoff_mm": 60.0
        }
    }

    # Example area population & safe sites for evacuation planning
    area_population_data = [
        {"area":"District A", "population":5000, "priority":1.0, "distance_matrix":{"s1":5,"s2":12}},
        {"area":"District B", "population":3000, "priority":0.8, "distance_matrix":{"s1":15,"s2":6}},
        {"area":"District C", "population":2000, "priority":0.7, "distance_matrix":{"s1":20,"s2":8}}
    ]
    safe_sites = [
        {"site_id":"s1","capacity":6000,"location":"School Ground A"},
        {"site_id":"s2","capacity":3000,"location":"Community Hall B"}
    ]

    # Schedule meeting
    meeting = coordinator.schedule_meeting(mock_risk_data, "Sindh", affected_provinces=["Sindh"])
    # Send notices
    notices = coordinator.send_meeting_notices(meeting)
    # Plan evacuation
    evac_plan = coordinator.plan_evacuation(area_population_data, safe_sites)
    # Generate dashboard + sitrep
    live_data = {
        "evacuees": evac_plan["total_evacuated"],
        "active_shelters": sum(1 for s in safe_sites if s["capacity"] < 999999),  # mock
        "blocked_roads": 5,
        "rescue_teams": 12
    }
    dashboard = coordinator.generate_meeting_dashboard(meeting, live_data)
    sitrep = coordinator.generate_sitrep(live_data)

    # Track meeting outcomes (mock)
    outcomes = {
        "decisions": ["Deploy 10 additional rescue teams", "Open two more shelters"],
        "action_items": ["NDMA to coordinate with PDMA within 2 hours"],
        "next_review_date": (datetime.now() + timedelta(hours=12)).isoformat(),
        "delays_reported": False,
        "resource_gaps": True
    }
    coordinator.track_meeting_outcomes(meeting, outcomes)

    # Print concise report
    print(f"Meeting: {meeting.meeting_id} scheduled at {meeting.datetime_scheduled}")
    print(f"Participants: {len(meeting.participants)}")
    print(f"Notices logged: {sum(notices.values())}/{len(notices)}")
    print(f"Evacuation coverage: {evac_plan['evacuation_coverage']*100:.1f}% ({evac_plan['total_evacuated']}/{evac_plan['total_population']})")
    print("Executive Summary:", dashboard["executive_summary"])
    print("SitRep snapshot:", sitrep["summary"])

if __name__ == "__main__":
    main()

