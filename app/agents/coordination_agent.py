

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

