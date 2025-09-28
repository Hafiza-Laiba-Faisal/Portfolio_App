"""
flood_orchestrator.py

Flask-based orchestration system for Pakistan Flood Management AI Agents
Integrates: Data Collection → Prediction → Meeting Coordination → Evacuation Planning
"""

from flask import Flask, request, jsonify, render_template_string
import json
import logging
from datetime import datetime
import threading
import time

# Import your existing agents
from data_agent import collect_flood_data, PAKISTAN_AREAS
from prediction_agent import predict_flood_risk
from flood_agentic_system import FloodMeetingCoordinator
from evacuation_agent import EvacuationAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'flood-management-pakistan-2024'

class FloodOrchestrator:
    def __init__(self):
        self.meeting_coordinator = FloodMeetingCoordinator()
        self.evacuation_agent = EvacuationAgent()
        self.active_assessments = {}
        self.alert_history = []
        
    def run_complete_assessment(self, area_key: str, custom_bbox=None):
        """Complete flood assessment pipeline"""
        assessment_id = f"ASSESS_{int(time.time())}"
        
        try:
            # Step 1: Data Collection
            logging.info(f"[{assessment_id}] Starting data collection for {area_key}")
            
            if custom_bbox:
                bbox = custom_bbox
                area_name = "Custom Area"
            else:
                if area_key not in PAKISTAN_AREAS:
                    raise ValueError(f"Unknown area: {area_key}")
                bbox = PAKISTAN_AREAS[area_key]["bbox"]
                area_name = PAKISTAN_AREAS[area_key]["name"]
            
            # Collect flood data
            flood_data = collect_flood_data(bbox, area_name)
            
            # Step 2: Risk Prediction
            logging.info(f"[{assessment_id}] Running risk prediction")
            prediction_result = predict_flood_risk(flood_data)
            
            # Step 3: Meeting Coordination (if needed)
            meeting = None
            if prediction_result["probability_score"] > 0.3:  # threshold for meeting
                logging.info(f"[{assessment_id}] Scheduling coordination meeting")
                meeting = self.meeting_coordinator.schedule_meeting(
                    prediction_result, area_name
                )
                # Send notices
                notice_results = self.meeting_coordinator.send_meeting_notices(meeting)
                
            # Step 4: Evacuation Planning (if HIGH risk)
            evacuation_plan = None
            if prediction_result["risk_level"] == "HIGH":
                logging.info(f"[{assessment_id}] Planning evacuation")
                evacuation_plan = self._generate_evacuation_plan(bbox, area_name)
            
            # Compile complete assessment
            complete_assessment = {
                "assessment_id": assessment_id,
                "timestamp": datetime.now().isoformat(),
                "area_name": area_name,
                "bbox": bbox,
                "data_collection": {
                    "status": "complete",
                    "sources_count": len(flood_data.get("data_sources", {})),
                    "weather_forecast": flood_data.get("data_sources", {}).get("weather", {}).get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [])
                },
                "risk_prediction": prediction_result,
                "meeting_coordination": {
                    "meeting_scheduled": meeting is not None,
                    "meeting_id": meeting.meeting_id if meeting else None,
                    "participants_count": len(meeting.participants) if meeting else 0,
                    "meeting_time": meeting.datetime_scheduled.isoformat() if meeting else None
                },
                "evacuation_planning": evacuation_plan,
                "recommendations": self._generate_integrated_recommendations(prediction_result, meeting, evacuation_plan)
            }
            
            # Store active assessment
            self.active_assessments[assessment_id] = complete_assessment
            
            # Add to alert history if significant
            if prediction_result["probability_score"] > 0.5:
                self.alert_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "area": area_name,
                    "risk_level": prediction_result["risk_level"],
                    "probability": prediction_result["probability_score"],
                    "assessment_id": assessment_id
                })
            
            logging.info(f"[{assessment_id}] Complete assessment finished")
            return complete_assessment
            
        except Exception as e:
            logging.error(f"[{assessment_id}] Assessment failed: {e}")
            return {
                "assessment_id": assessment_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_evacuation_plan(self, bbox, area_name):
        """Generate evacuation plan for high-risk areas"""
        # Mock population data based on bbox size
        area_km2 = self._calculate_bbox_area(bbox)
        estimated_population = int(area_km2 * 1000)  # rough estimate
        
        # Create mock evacuation areas
        area_population_data = [
            {
                "area": f"{area_name} Central",
                "population": int(estimated_population * 0.4),
                "priority": 1.0,
                "distance_matrix": {"camp1": 5, "camp2": 12}
            },
            {
                "area": f"{area_name} Outskirts", 
                "population": int(estimated_population * 0.6),
                "priority": 0.7,
                "distance_matrix": {"camp1": 15, "camp2": 8}
            }
        ]
        
        # Mock safe sites
        safe_sites = [
            {"site_id": "camp1", "capacity": int(estimated_population * 0.6), "location": f"{area_name} School Ground"},
            {"site_id": "camp2", "capacity": int(estimated_population * 0.4), "location": f"{area_name} Community Center"}
        ]
        
        evacuation_plan = self.meeting_coordinator.plan_evacuation(area_population_data, safe_sites)
        return evacuation_plan
    
    def _calculate_bbox_area(self, bbox):
        """Calculate approximate area in km2"""
        import math
        lat1, lon1, lat2, lon2 = bbox
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        avg_lat = (lat1 + lat2) / 2
        area = abs(dlat * 111.32 * dlon * 111.32 * math.cos(math.radians(avg_lat)))
        return max(area, 1.0)
    
    def _generate_integrated_recommendations(self, prediction, meeting, evacuation_plan):
        """Generate integrated recommendations from all agents"""
        recommendations = prediction.get("recommendations", [])
        
        if meeting:
            recommendations.append(f"Coordination meeting scheduled for {meeting.datetime_scheduled.strftime('%Y-%m-%d %H:%M')} with {len(meeting.participants)} stakeholders")
        
        if evacuation_plan:
            coverage = evacuation_plan.get("evacuation_coverage", 0) * 100
            recommendations.append(f"Evacuation plan prepared with {coverage:.1f}% population coverage")
            if evacuation_plan.get("unallocated"):
                recommendations.append("Additional shelter capacity required for complete evacuation")
        
        return recommendations

# Initialize orchestrator
orchestrator = FloodOrchestrator()

# Flask Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Pakistan Flood Management System</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #3498db; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .status { padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
        .status-low { background: #27ae60; }
        .status-medium { background: #f39c12; }
        .status-high { background: #e74c3c; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 Pakistan Flood Management AI System</h1>
            <p>Integrated Data Collection • Risk Prediction • Meeting Coordination • Evacuation Planning</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Quick Assessment</h2>
                <p>Start flood risk assessment for Pakistani cities:</p>
                <button class="btn btn-primary" onclick="runAssessment('karachi')">Karachi</button>
                <button class="btn btn-primary" onclick="runAssessment('lahore')">Lahore</button>
                <button class="btn btn-primary" onclick="runAssessment('islamabad')">Islamabad</button>
                <button class="btn btn-primary" onclick="runAssessment('rawalpindi')">Rawalpindi</button>
            </div>
            
            <div class="card">
                <h2>System Status</h2>
                <p><strong>Active Assessments:</strong> <span id="active-count">{{ active_count }}</span></p>
                <p><strong>Recent Alerts:</strong> <span id="alert-count">{{ alert_count }}</span></p>
                <p><strong>Last Update:</strong> {{ timestamp }}</p>
            </div>
        </div>
        
        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><code>POST /assess/{area}</code> - Run complete assessment</li>
                <li><code>GET /status</code> - System status</li>
                <li><code>GET /alerts</code> - Recent alerts</li>
                <li><code>POST /evacuate</code> - Evacuation guidance</li>
            </ul>
        </div>
        
        <div class="card" id="results">
            <h2>Assessment Results</h2>
            <p>Results will appear here after running an assessment...</p>
        </div>
    </div>

    <script>
        function runAssessment(area) {
            document.getElementById('results').innerHTML = '<h2>Assessment Results</h2><p>Running assessment for ' + area + '...</p>';
            
            fetch('/assess/' + area, {method: 'POST'})
                .then(response => response.json())
                .then(data => displayResults(data))
                .catch(error => {
                    document.getElementById('results').innerHTML = '<h2>Assessment Results</h2><p>Error: ' + error + '</p>';
                });
        }
        
        function displayResults(data) {
            let html = '<h2>Assessment Results - ' + data.area_name + '</h2>';
            
            if (data.status === 'error') {
                html += '<p style="color: red;">Error: ' + data.error + '</p>';
            } else {
                let riskClass = 'status-' + data.risk_prediction.risk_level.toLowerCase();
                html += '<p><span class="status ' + riskClass + '">' + data.risk_prediction.risk_level + ' RISK</span></p>';
                html += '<p><strong>Probability Score:</strong> ' + (data.risk_prediction.probability_score * 100).toFixed(1) + '%</p>';
                html += '<p><strong>7-Day Precipitation:</strong> ' + data.risk_prediction.numbers.forecast_precip_next7_mm.toFixed(1) + ' mm</p>';
                
                if (data.meeting_coordination.meeting_scheduled) {
                    html += '<p><strong>Meeting Scheduled:</strong> ' + data.meeting_coordination.meeting_time + '</p>';
                    html += '<p><strong>Participants:</strong> ' + data.meeting_coordination.participants_count + '</p>';
                }
                
                if (data.evacuation_planning) {
                    let coverage = (data.evacuation_planning.evacuation_coverage * 100).toFixed(1);
                    html += '<p><strong>Evacuation Coverage:</strong> ' + coverage + '%</p>';
                }
                
                html += '<h3>Recommendations:</h3><ul>';
                data.recommendations.forEach(rec => {
                    html += '<li>' + rec + '</li>';
                });
                html += '</ul>';
                
                html += '<p><strong>Urdu Summary:</strong> ' + data.risk_prediction.urdu_summary + '</p>';
            }
            
            document.getElementById('results').innerHTML = html;
        }
        
        // Auto-refresh status every 30 seconds
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('active-count').textContent = data.active_assessments;
                    document.getElementById('alert-count').textContent = data.recent_alerts;
                });
        }, 30000);
    </script>
</body>
</html>
    """
    
    return render_template_string(dashboard_template, 
        active_count=len(orchestrator.active_assessments),
        alert_count=len(orchestrator.alert_history),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S PKT')
    )

@app.route('/assess/<area_key>', methods=['POST'])
def run_assessment(area_key):
    """Run complete flood assessment for specified area"""
    try:
        # Run in background thread for better response time
        def background_assessment():
            return orchestrator.run_complete_assessment(area_key)
        
        result = background_assessment()  # For demo, run synchronously
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Assessment endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/assess/custom', methods=['POST'])
def run_custom_assessment():
    """Run assessment for custom bbox"""
    try:
        data = request.json
        bbox = data.get('bbox')  # [min_lat, min_lon, max_lat, max_lon]
        
        if not bbox or len(bbox) != 4:
            return jsonify({"error": "Invalid bbox format"}), 400
        
        result = orchestrator.run_complete_assessment("custom", bbox)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify({
        "active_assessments": len(orchestrator.active_assessments),
        "recent_alerts": len([a for a in orchestrator.alert_history if 
                            (datetime.now() - datetime.fromisoformat(a["timestamp"])).days < 1]),
        "total_meetings": len(orchestrator.meeting_coordinator.meeting_history),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify({
        "alerts": orchestrator.alert_history[-10:],  # Last 10 alerts
        "count": len(orchestrator.alert_history)
    })

@app.route('/evacuate', methods=['POST'])
def evacuation_guidance():
    """Provide evacuation guidance"""
    try:
        data = request.json
        location = data.get('location')
        
        if not location:
            return jsonify({"error": "Location required"}), 400
        
        # Mock evacuation guidance (integrate with your evacuation_agent)
        guidance = {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "nearest_camps": "Use evacuation_agent.guide_user() method",
            "emergency_contacts": {
                "NDMA": "051-111222333",
                "Emergency": "112",
                "Rescue 1122": "1122"
            }
        }
        
        return jsonify(guidance)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/meeting/<meeting_id>')
def get_meeting_details(meeting_id):
    """Get meeting details"""
    # Find meeting in history or active meetings
    for meeting_record in orchestrator.meeting_coordinator.meeting_history:
        if meeting_record.get("meeting_id") == meeting_id:
            return jsonify(meeting_record)
    
    for meeting in orchestrator.meeting_coordinator.active_meetings:
        if meeting.meeting_id == meeting_id:
            return jsonify({
                "meeting_id": meeting.meeting_id,
                "stage": meeting.flood_stage.value,
                "scheduled_time": meeting.datetime_scheduled.isoformat(),
                "participants": [p.name for p in meeting.participants],
                "agenda": meeting.agenda_items
            })
    
    return jsonify({"error": "Meeting not found"}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_collection": "operational",
            "prediction_engine": "operational", 
            "meeting_coordinator": "operational",
            "evacuation_agent": "operational"
        }
    })

if __name__ == '__main__':
    print("🌊 Starting Pakistan Flood Management System...")
    print("📍 Dashboard: http://localhost:5000")
    print("📊 API Health: http://localhost:5000/health")
    print("⚡ Quick Assessment: POST http://localhost:5000/assess/karachi")
    
    app.run(debug=True, host='0.0.0.0', port=5000)