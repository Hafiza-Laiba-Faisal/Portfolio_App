# import os
# import json
# import sqlite3
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
# from dotenv import load_dotenv
# from app.new_agents.hydro_met.data_collection import collect_flood_data, PAKISTAN_AREAS
# from app.new_agents.hydro_met.prediction import predict_flood_risk
# from openai import AsyncOpenAI
# from pydantic import BaseModel
# import asyncio
# # Load environment variables
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# # Configure OpenAI client for Gemini
# client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# # Validation Models
# class FloodInputSanitizer(BaseModel):
#     is_valid: bool
#     reason: str | None = None

# class FloodOutputSanitizer(BaseModel):
#     is_valid: bool
#     reason: str | None = None

# # HydroMetTool
# class HydroMetInput(BaseModel):
#     choice: str
#     bbox: list[float] | None = None
#     area_name: str | None = None

# class HydroMetOutput(BaseModel):
#     prediction_result: dict
#     chart_config: dict
#     alert_message: str
#     visualization_files: list[str]

# class HydroMetTool:
#     @staticmethod
#     async def validate_input(input_str: str) -> FloodInputSanitizer:
#         """Validate input using Gemini model"""
#         try:
#             prompt = (
#                 f"Check if the input is valid for flood risk analysis. The input should either be a number (1-9) or a comma-separated string of four coordinates (min_lat, min_lon, max_lat, max_lon) within reasonable bounds (latitudes 0-90, longitudes 0-180). Input: {input_str}\n"
#                 "Return a JSON object: {'is_valid': bool, 'reason': str | null}"
#             )
#             response = await client.chat.completions.create(
#                 model="gemini-2.5-flash",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
#             result = json.loads(response.choices[0].message.content)
#             return FloodInputSanitizer(**result)
#         except Exception as e:
#             return FloodInputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     @staticmethod
#     async def validate_output(output: dict) -> FloodOutputSanitizer:
#         """Validate output using Gemini model"""
#         try:
#             output_str = json.dumps(output)
#             prompt = (
#                 f"Check if the flood prediction output is valid. The output should be a JSON object with 'risk_level' (LOW, MEDIUM, HIGH), 'probability_score' (0-1), and numerical fields (e.g., forecast_precip_next1_mm, curve_number) without NaN or null values. Output: {output_str}\n"
#                 "Return a JSON object: {'is_valid': bool, 'reason': str | null}"
#             )
#             response = await client.chat.completions.create(
#                 model="gemini-2.5-flash",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
#             result = json.loads(response.choices[0].message.content)
#             return FloodOutputSanitizer(**result)
#         except Exception as e:
#             return FloodOutputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     @staticmethod
#     async def send_to_cortex(prediction_result: dict, alert_message: str, visualization_files: list[str]):
#         """
#         Send processed flood data to Cortex agent.
#         Replace the print statements with actual Cortex integration.
#         """
#         print("\n➡️ Sending data to Cortex agent...")
#         payload = {
#             "prediction_result": prediction_result,
#             "alert_message": alert_message,
#             "visualization_files": visualization_files
#         }
#         # Example integration (replace with actual Cortex agent API):
#         # await cortex_agent.receive(payload)
#         print(json.dumps(payload, indent=2, ensure_ascii=False))
#         print("✅ Data sent to Cortex.")

#     @staticmethod
#     async def run(input_data: HydroMetInput) -> HydroMetOutput:
#         """Run flood risk analysis (SQLite-free, Cortex-ready)"""
#         try:
#             # --- Input validation ---
#             input_str = input_data.choice
#             if input_data.choice == "custom":
#                 input_str = ",".join(map(str, input_data.bbox or []))
#             input_validation = await HydroMetTool.validate_input(input_str)
#             if not input_validation.is_valid:
#                 raise ValueError(f"Invalid input: {input_validation.reason}")

#             # --- Process input ---
#             if input_data.choice == "custom":
#                 if not input_data.bbox or len(input_data.bbox) != 4:
#                     raise ValueError("Custom area requires a valid bounding box (min_lat, min_lon, max_lat, max_lon)")
#                 bbox = input_data.bbox
#                 name = input_data.area_name or "Custom Area"
#             else:
#                 choice = int(input_data.choice)
#                 if 1 <= choice <= len(PAKISTAN_AREAS):
#                     area_key = list(PAKISTAN_AREAS.keys())[choice-1]
#                     area_config = PAKISTAN_AREAS[area_key]
#                     bbox = area_config["bbox"]
#                     name = area_config["name"]
#                 else:
#                     raise ValueError(f"Invalid choice: {choice}. Must be 1-{len(PAKISTAN_AREAS)+1}")

#             # --- Collect data ---
#             flood_data = collect_flood_data(bbox, name)

#             # --- Run prediction ---
#             prediction_result = predict_flood_risk(flood_data)

#             # --- Validate output ---
#             output_validation = await HydroMetTool.validate_output(prediction_result)
#             if not output_validation.is_valid:
#                 raise ValueError(f"Invalid output: {output_validation.reason}")

#             # --- Generate alert message ---
#             risk_level = prediction_result["risk_level"]
#             urdu_summary = prediction_result["urdu_summary"]
#             recommendations = prediction_result["recommendations"]
#             probability_score = prediction_result["probability_score"]
#             alert_message = (
#                 f"Flood Risk Alert for {name}:\n"
#                 f"Risk Level: {risk_level}\n"
#                 f"Probability Score: {probability_score:.2f}\n"
#                 f"Urdu Summary: {urdu_summary}\n"
#                 f"Recommendations:\n" + "\n".join(recommendations)
#             )
#             if risk_level in ["MEDIUM", "HIGH"]:
#                 alert_message += f"\nTriggering early-warning notifications to downstream AI agents for {risk_level} risk in {name}."

#             # --- Generate Chart.js config ---
#             precip = flood_data["data_sources"]["weather"].get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [0]*7)
#             days = list(range(1, len(precip)+1))
#             chart_config = {
#                 "type": "line",
#                 "data": {"labels": days, "datasets":[{"label":"Daily Precipitation (mm)","data":precip,"borderColor":"#1e90ff","backgroundColor":"rgba(30, 144, 255, 0.2)","fill":True,"tension":0.3}]},
#                 "options": {"responsive": True, "plugins":{"title":{"display": True,"text":f"7-Day Precipitation Forecast for {name} ({flood_data['meta']['datetime']})"}},"scales":{"x":{"title":{"display": True,"text":"Day"}},"y":{"title":{"display": True,"text":"Precipitation (mm)"},"beginAtZero": True}}}
#             }

#             # --- Generate visualizations ---
#             output_dir = 'visualizations'
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             timestamp = flood_data['meta']['datetime'].replace(':','-')
#             # Precipitation chart
#             plt.figure(figsize=(10,5))
#             plt.plot(range(1,8), [max(0,x) for x in [prediction_result['numbers']['forecast_precip_next1_mm'],
#                                                      prediction_result['numbers']['forecast_precip_next3_mm']-prediction_result['numbers']['forecast_precip_next1_mm'],
#                                                      prediction_result['numbers']['forecast_precip_next7_mm']-prediction_result['numbers']['forecast_precip_next3_mm'],0,0,0,0][:7]],
#                      marker='o', linestyle='-', color='b')
#             plt.title(f'7-Day Precipitation Forecast for {name} ({flood_data["meta"]["datetime"]})')
#             plt.xlabel('Day')
#             plt.ylabel('Precipitation (mm)')
#             plt.grid(True)
#             plt.tight_layout()
#             precip_filename = f'{output_dir}/precipitation_forecast_{name.replace(" ","_")}_{timestamp}.png'
#             plt.savefig(precip_filename)
#             plt.close()
#             # Key metrics
#             metrics = ['Runoff (mm)','Peak Discharge (m³/s)','Time to Peak (hr)','AMC (0-1)','Anomaly Score (0-1)']
#             viz_values = [prediction_result['numbers']['estimated_runoff_mm'],
#                           prediction_result['numbers']['estimated_peak_discharge_m3s'],
#                           prediction_result['numbers']['estimated_time_to_peak_hours'],
#                           prediction_result['diagnostics']['amc_info']['amc'],
#                           float(prediction_result['key_factors'][4].split(': ')[1])]
#             plt.figure(figsize=(12,6))
#             bars = plt.bar(metrics,viz_values,color=['blue','green','orange','purple','red'])
#             plt.title(f'Key Flood Forecasting Metrics for {name} ({flood_data["meta"]["datetime"]})')
#             plt.ylabel('Value')
#             plt.grid(axis='y')
#             for bar, value in zip(bars, viz_values):
#                 plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=10)
#             plt.tight_layout()
#             metrics_filename = f'{output_dir}/key_metrics_{name.replace(" ","_")}_{timestamp}.png'
#             plt.savefig(metrics_filename)
#             plt.close()
#             visualization_files = [precip_filename, metrics_filename]

#             # --- Send data to Cortex ---
#             await HydroMetTool.send_to_cortex(prediction_result, alert_message, visualization_files)

#             return HydroMetOutput(
#                 prediction_result=prediction_result,
#                 chart_config=chart_config,
#                 alert_message=alert_message,
#                 visualization_files=visualization_files
#             )

#         except Exception as e:
#             raise ValueError(f"HydroMetTool error: {str(e)}")

#     @staticmethod
#     async def validate_input(input_str: str) -> FloodInputSanitizer:
#         """Validate input using Gemini model"""
#         try:
#             prompt = (
#                 f"Check if the input is valid for flood risk analysis. The input should either be a number (1-9) or a comma-separated string of four coordinates (min_lat, min_lon, max_lat, max_lon) within reasonable bounds (latitudes 0-90, longitudes 0-180). Input: {input_str}\n"
#                 "Return a JSON object: {'is_valid': bool, 'reason': str | null}"
#             )
#             response = await client.chat.completions.create(
#                 model="gemini-2.5-flash",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
#             result = json.loads(response.choices[0].message.content)
#             return FloodInputSanitizer(**result)
#         except Exception as e:
#             return FloodInputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     @staticmethod
#     async def validate_output(output: dict) -> FloodOutputSanitizer:
#         """Validate output using Gemini model"""
#         try:
#             output_str = json.dumps(output)
#             prompt = (
#                 f"Check if the flood prediction output is valid. The output should be a JSON object with 'risk_level' (LOW, MEDIUM, HIGH), 'probability_score' (0-1), and numerical fields (e.g., forecast_precip_next1_mm, curve_number) without NaN or null values. Output: {output_str}\n"
#                 "Return a JSON object: {'is_valid': bool, 'reason': str | null}"
#             )
#             response = await client.chat.completions.create(
#                 model="gemini-2.5-flash",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"}
#             )
#             result = json.loads(response.choices[0].message.content)
#             return FloodOutputSanitizer(**result)
#         except Exception as e:
#             return FloodOutputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     @staticmethod
#     async def run(input_data: HydroMetInput) -> HydroMetOutput:
#         """Run flood risk analysis"""
#         try:
#             # Validate input
#             input_str = input_data.choice
#             if input_data.choice == "custom":
#                 input_str = ",".join(map(str, input_data.bbox or []))
#             input_validation = await HydroMetTool.validate_input(input_str)
#             if not input_validation.is_valid:
#                 raise ValueError(f"Invalid input: {input_validation.reason}")

#             # Process input
#             if input_data.choice == "custom":
#                 if not input_data.bbox or len(input_data.bbox) != 4:
#                     raise ValueError("Custom area requires a valid bounding box (min_lat, min_lon, max_lat, max_lon)")
#                 bbox = input_data.bbox
#                 name = input_data.area_name or "Custom Area"
#             else:
#                 choice = int(input_data.choice)
#                 if 1 <= choice <= len(PAKISTAN_AREAS):
#                     area_key = list(PAKISTAN_AREAS.keys())[choice-1]
#                     area_config = PAKISTAN_AREAS[area_key]
#                     bbox = area_config["bbox"]
#                     name = area_config["name"]
#                 else:
#                     raise ValueError(f"Invalid choice: {choice}. Must be 1-{len(PAKISTAN_AREAS)+1}")

#             # Collect data
#             flood_data = collect_flood_data(bbox, name)

#             # Run prediction
#             prediction_result = predict_flood_risk(flood_data)

#             # Validate output
#             output_validation = await HydroMetTool.validate_output(prediction_result)
#             if not output_validation.is_valid:
#                 raise ValueError(f"Invalid output: {output_validation.reason}")

#             # Generate alert message
#             risk_level = prediction_result["risk_level"]
#             urdu_summary = prediction_result["urdu_summary"]
#             recommendations = prediction_result["recommendations"]
#             probability_score = prediction_result["probability_score"]
#             alert_message = (
#                 f"Flood Risk Alert for {name}:\n"
#                 f"Risk Level: {risk_level}\n"
#                 f"Probability Score: {probability_score:.2f}\n"
#                 f"Urdu Summary: {urdu_summary}\n"
#                 f"Recommendations:\n" + "\n".join(recommendations)
#             )
#             if risk_level in ["MEDIUM", "HIGH"]:
#                 alert_message += f"\nTriggering early-warning notifications to downstream AI agents for {risk_level} risk in {name}."

#             # Create Chart.js configuration
#             precip = flood_data["data_sources"]["weather"].get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [0] * 7)
#             days = list(range(1, len(precip) + 1))
#             chart_config = {
#                 "type": "line",
#                 "data": {
#                     "labels": days,
#                     "datasets": [{
#                         "label": "Daily Precipitation (mm)",
#                         "data": precip,
#                         "borderColor": "#1e90ff",
#                         "backgroundColor": "rgba(30, 144, 255, 0.2)",
#                         "fill": True,
#                         "tension": 0.3
#                     }]
#                 },
#                 "options": {
#                     "responsive": True,
#                     "plugins": {
#                         "title": {
#                             "display": True,
#                             "text": f"7-Day Precipitation Forecast for {name} ({flood_data['meta']['datetime']})"
#                         }
#                     },
#                     "scales": {
#                         "x": {
#                             "title": {
#                                 "display": True,
#                                 "text": "Day"
#                             }
#                         },
#                         "y": {
#                             "title": {
#                                 "display": True,
#                                 "text": "Precipitation (mm)"
#                             },
#                             "beginAtZero": True
#                         }
#                     }
#                 }
#             }

#             # Store data in SQLite
#             conn = sqlite3.connect('flood_data.db')
#             cursor = conn.cursor()
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS Meta (
#                 id INTEGER PRIMARY KEY,
#                 area_name TEXT,
#                 bbox TEXT,
#                 center TEXT,
#                 timestamp INTEGER,
#                 datetime TEXT,
#                 agent TEXT,
#                 version TEXT,
#                 role TEXT,
#                 completion_time INTEGER
#             )
#             ''')
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS Predictions (
#                 id INTEGER PRIMARY KEY,
#                 meta_id INTEGER,
#                 risk_level TEXT,
#                 probability_score REAL,
#                 forecast_precip_next1_mm REAL,
#                 forecast_precip_next3_mm REAL,
#                 forecast_precip_next7_mm REAL,
#                 curve_number REAL,
#                 estimated_runoff_mm REAL,
#                 estimated_peak_discharge_m3s REAL,
#                 estimated_time_to_peak_hours REAL,
#                 key_factors TEXT,
#                 recommendations TEXT,
#                 urdu_summary TEXT,
#                 diagnostics TEXT,
#                 FOREIGN KEY (meta_id) REFERENCES Meta(id)
#             )
#             ''')
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS DataSources (
#                 id INTEGER PRIMARY KEY,
#                 meta_id INTEGER,
#                 weather TEXT,
#                 elevation TEXT,
#                 water_bodies TEXT,
#                 infrastructure TEXT,
#                 satellite TEXT,
#                 soil TEXT,
#                 landuse TEXT,
#                 river TEXT,
#                 historical TEXT,
#                 satellite_flood TEXT,
#                 FOREIGN KEY (meta_id) REFERENCES Meta(id)
#             )
#             ''')
#             meta = flood_data['meta']
#             cursor.execute('''
#             INSERT INTO Meta (area_name, bbox, center, timestamp, datetime, agent, version, role, completion_time)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta['area_name'],
#                 json.dumps(meta['bbox']),
#                 json.dumps(meta['center']),
#                 meta['timestamp'],
#                 meta['datetime'],
#                 meta['agent'],
#                 meta['version'],
#                 meta['role'],
#                 meta['completion_time']
#             ))
#             meta_id = cursor.lastrowid
#             data_sources = flood_data['data_sources']
#             cursor.execute('''
#             INSERT INTO DataSources (meta_id, weather, elevation, water_bodies, infrastructure, satellite, soil, landuse, river, historical, satellite_flood)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta_id,
#                 json.dumps(data_sources['weather']),
#                 json.dumps(data_sources['elevation']),
#                 json.dumps(data_sources['water_bodies']),
#                 json.dumps(data_sources['infrastructure']),
#                 json.dumps(data_sources['satellite']),
#                 json.dumps(data_sources['soil']),
#                 json.dumps(data_sources['landuse']),
#                 json.dumps(data_sources['river']),
#                 json.dumps(data_sources['historical']),
#                 json.dumps(data_sources['satellite_flood'])
#             ))
#             pred_numbers = prediction_result['numbers']
#             pred_diagnostics = prediction_result['diagnostics']
#             cursor.execute('''
#             INSERT INTO Predictions (meta_id, risk_level, probability_score, forecast_precip_next1_mm, forecast_precip_next3_mm, forecast_precip_next7_mm,
#             curve_number, estimated_runoff_mm, estimated_peak_discharge_m3s, estimated_time_to_peak_hours, key_factors, recommendations, urdu_summary, diagnostics)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta_id,
#                 prediction_result['risk_level'],
#                 prediction_result['probability_score'],
#                 pred_numbers['forecast_precip_next1_mm'],
#                 pred_numbers['forecast_precip_next3_mm'],
#                 pred_numbers['forecast_precip_next7_mm'],
#                 pred_numbers['curve_number'],
#                 pred_numbers['estimated_runoff_mm'],
#                 pred_numbers['estimated_peak_discharge_m3s'],
#                 pred_numbers['estimated_time_to_peak_hours'],
#                 json.dumps(prediction_result['key_factors']),
#                 json.dumps(prediction_result['recommendations']),
#                 prediction_result['urdu_summary'],
#                 json.dumps(pred_diagnostics)
#             ))
#             conn.commit()
#             conn.close()
#             print(f"Data stored in SQLite database 'flood_data.db' for {meta['area_name']} at {meta['datetime']}")

#             # Generate visualizations
#             viz_data = {
#                 'area_name': name,
#                 'datetime': flood_data['meta']['datetime'],
#                 'risk_level': prediction_result['risk_level'],
#                 'probability_score': prediction_result['probability_score'],
#                 'precip_next1': prediction_result['numbers']['forecast_precip_next1_mm'],
#                 'precip_next3': prediction_result['numbers']['forecast_precip_next3_mm'],
#                 'precip_next7': prediction_result['numbers']['forecast_precip_next7_mm'],
#                 'curve_number': prediction_result['numbers']['curve_number'],
#                 'runoff': prediction_result['numbers']['estimated_runoff_mm'],
#                 'peak_discharge': prediction_result['numbers']['estimated_peak_discharge_m3s'],
#                 'time_to_peak': prediction_result['numbers']['estimated_time_to_peak_hours'],
#                 'amc': prediction_result['diagnostics']['amc_info']['amc'],
#                 'anomaly_score': float(prediction_result['key_factors'][4].split(': ')[1]),
#                 'water_count': prediction_result['diagnostics']['water_count']
#             }
#             output_dir = 'visualizations'
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             area_name = viz_data['area_name']
#             timestamp = viz_data['datetime'].replace(':', '-')
#             timestamp_display = datetime.strptime(viz_data['datetime'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
#             precip_days = [
#                 viz_data['precip_next1'],
#                 viz_data['precip_next3'] - viz_data['precip_next1'],
#                 viz_data['precip_next7'] - viz_data['precip_next3'],
#                 0, 0, 0, 0
#             ]
#             precip_days = np.array([max(0, x) for x in precip_days[:7]])
#             plt.figure(figsize=(10, 5))
#             plt.plot(range(1, 8), precip_days, marker='o', linestyle='-', color='b')
#             plt.title(f'7-Day Precipitation Forecast for {area_name} ({timestamp_display})')
#             plt.xlabel('Day')
#             plt.ylabel('Precipitation (mm)')
#             plt.grid(True)
#             plt.tight_layout()
#             precip_filename = f'{output_dir}/precipitation_forecast_{area_name.replace(" ", "_")}_{timestamp}.png'
#             plt.savefig(precip_filename)
#             plt.close()
#             metrics = ['Runoff (mm)', 'Peak Discharge (m³/s)', 'Time to Peak (hr)', 'AMC (0-1)', 'Anomaly Score (0-1)']
#             values = [viz_data['runoff'], viz_data['peak_discharge'], viz_data['time_to_peak'], viz_data['amc'], viz_data['anomaly_score']]
#             plt.figure(figsize=(12, 6))
#             bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple', 'red'])
#             plt.title(f'Key Flood Forecasting Metrics for {area_name} ({timestamp_display})')
#             plt.ylabel('Value')
#             plt.grid(axis='y')
#             for bar, value in zip(bars, values):
#                 plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}',
#                          ha='center', va='bottom', fontsize=10)
#             plt.tight_layout()
#             metrics_filename = f'{output_dir}/key_metrics_{area_name.replace(" ", "_")}_{timestamp}.png'
#             plt.savefig(metrics_filename)
#             plt.close()
#             visualization_files = [precip_filename, metrics_filename]

#             return HydroMetOutput(
#                 prediction_result=prediction_result,
#                 chart_config=chart_config,
#                 alert_message=alert_message,
#                 visualization_files=visualization_files
#             )

#         except Exception as e:
#             raise ValueError(f"HydroMetTool error: {str(e)}")

# # Main Execution
# if __name__ == "__main__":
#     try:
#         print("\n🌍 Select target area for flood risk analysis:\n")
#         for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
#             print(f"{i}. {val['name']} ({key})")
#         print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")
        
#         choice_input = input("\nEnter choice number: ").strip()
#         tool_input = None
#         if choice_input == str(len(PAKISTAN_AREAS)+1):
#             print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
#             bbox_input = input("Format: lat1,lon1,lat2,lon2 → ").strip()
#             area_name = input("Enter area name (optional, press Enter for 'Custom Area'): ").strip() or "Custom Area"
#             tool_input = HydroMetInput(choice="custom", bbox=list(map(float, bbox_input.split(","))), area_name=area_name)
#         else:
#             tool_input = HydroMetInput(choice=choice_input)
        
#         result = asyncio.run(HydroMetTool.run(tool_input))
        
#         print("\n📊 Prediction Results:")
#         print(json.dumps(result.prediction_result, indent=2, ensure_ascii=False))
        
#         print("\n📈 Precipitation Chart:")
#         print(json.dumps(result.chart_config, indent=2))
        
#         print("\n🚨 Alerts:")
#         print(result.alert_message)
        
#         print("\n📊 Visualizations Generated:")
#         for file in result.visualization_files:
#             print(f"- {file}")
            
#     except ValueError as e:
#         print(f"Error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")


# from agents import Agent

# hydro_met_agent = Agent(
#     name="HydroMetAgent",
#     instructions="You are a flood risk forecasting agent for Pakistan. Use HydroMetTool to predict and alert.",
#     model=None,  # agar LLM ka use karna ho to llm_model pass karo
#     tools=[HydroMetTool]
# )




# import os
# import json
# import sqlite3
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
# from dotenv import load_dotenv
# import logging
# try:
#     from app.new_agents.hydro_met.data_collection import collect_flood_data, PAKISTAN_AREAS
#     from app.new_agents.hydro_met.prediction import predict_flood_risk
# except ModuleNotFoundError:
#     logging.warning("Failed to import data_collection or prediction. Using mock implementations.")
#     PAKISTAN_AREAS = {
#         "karachi": {"name": "Karachi Metropolitan", "bbox": [24.8, 66.9, 25.2, 67.3]},
#         "lahore": {"name": "Lahore District", "bbox": [31.0, 74.0, 31.5, 74.5]},
#         "islamabad": {"name": "Islamabad Capital Territory", "bbox": [33.6, 73.0, 33.8, 73.2]},
#         "faisalabad": {"name": "Faisalabad District", "bbox": [31.3, 73.0, 31.5, 73.2]},
#         "rawalpindi": {"name": "Rawalpindi District", "bbox": [33.5, 73.0, 33.7, 73.2]},
#         "multan": {"name": "Multan District", "bbox": [30.1, 71.4, 30.3, 71.6]},
#         "peshawar": {"name": "Peshawar District", "bbox": [34.0, 71.5, 34.2, 71.7]},
#         "quetta": {"name": "Quetta District", "bbox": [30.1, 66.9, 30.3, 67.1]}
#     }
#     def collect_flood_data(bbox, name):
#         return {
#             "meta": {
#                 "area_name": name,
#                 "bbox": bbox,
#                 "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
#                 "timestamp": int(datetime.now().timestamp()),
#                 "datetime": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
#                 "agent": "HydroMetAgent",
#                 "version": "1.0",
#                 "role": "FloodRiskAnalysis",
#                 "completion_time": 0
#             },
#             "data_sources": {
#                 "weather": {"open_meteo": {"daily": {"precipitation_sum": [10, 20, 30, 40, 50, 60, 70]}}},
#                 "elevation": {}, "water_bodies": {}, "infrastructure": {}, "satellite": {},
#                 "soil": {}, "landuse": {}, "river": {}, "historical": {}, "satellite_flood": {}
#             }
#         }
#     def predict_flood_risk(flood_data):
#         return {
#             "risk_level": "MEDIUM",
#             "probability_score": 0.75,
#             "urdu_summary": f"{flood_data['meta']['area_name']} میں درمیانے درجے کا سیلاب کا خطرہ",
#             "recommendations": ["Monitor river levels", "Prepare evacuation plans"],
#             "numbers": {
#                 "forecast_precip_next1_mm": 10.0,
#                 "forecast_precip_next3_mm": 25.0,
#                 "forecast_precip_next7_mm": 50.0,
#                 "curve_number": 80.0,
#                 "estimated_runoff_mm": 15.0,
#                 "estimated_peak_discharge_m3s": 100.0,
#                 "estimated_time_to_peak_hours": 12.0
#             },
#             "diagnostics": {"amc_info": {"amc": 0.5}},
#             "key_factors": ["Precipitation: High", "Soil Moisture: Moderate", "Elevation: Low", "Land Use: Urban", "Anomaly Score: 0.8"]
#         }

# from pydantic import BaseModel, field_validator, ConfigDict
# from agents import Agent, function_tool, set_tracing_disabled, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# import asyncio
# import inspect

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
# logger = logging.getLogger(__name__)

# # Log Runner class details for debugging
# logger.debug(f"Runner class: {inspect.getmembers(Runner, predicate=inspect.isfunction)}")

# # Load environment variables
# load_dotenv()
# # Disable all tracing
# set_tracing_disabled(True)

# # Validation Models
# class FloodInputSanitizer(BaseModel):
#     is_valid: bool
#     reason: str | None = None
#     model_config = ConfigDict(extra="forbid")

# class FloodOutputSanitizer(BaseModel):
#     is_valid: bool
#     reason: str | None = None
#     model_config = ConfigDict(extra="forbid")

# # HydroMetInput and Output Models
# class HydroMetToolInput(BaseModel):
#     choice: str
#     bbox: list[float] | None = None
#     area_name: str | None = None
#     model_config = ConfigDict(extra="forbid")

#     @field_validator('bbox')
#     def validate_bbox(cls, v):
#         if v and len(v) != 4:
#             raise ValueError("Bounding box must contain exactly 4 values: min_lat, min_lon, max_lat, max_lon")
#         if v:
#             min_lat, min_lon, max_lat, max_lon = v
#             if not (0 <= min_lat <= 90 and 0 <= max_lat <= 90 and 0 <= min_lon <= 180 and 0 <= max_lon <= 180):
#                 raise ValueError("Coordinates out of valid range: latitudes (0-90), longitudes (0-180)")
#         return v

# class HydroMetToolOutput(BaseModel):
#     prediction_result: dict
#     chart_config: dict
#     alert_message: str
#     visualization_files: list[str]
#     model_config = ConfigDict(extra="forbid")

# # HydroMetTool Class
# class HydroMetTool:
#     async def validate_input(self, input_str: str) -> FloodInputSanitizer:
#         """Validate input using simple checks (no API calls)"""
#         try:
#             logger.debug(f"Validating input: {input_str}")
#             if input_str == "custom":
#                 return FloodInputSanitizer(is_valid=True, reason="Custom input accepted")
#             try:
#                 choice = int(input_str)
#                 if 1 <= choice <= len(PAKISTAN_AREAS):
#                     return FloodInputSanitizer(is_valid=True, reason=None)
#                 else:
#                     return FloodInputSanitizer(is_valid=False, reason=f"Choice {choice} out of range (1-{len(PAKISTAN_AREAS)})")
#             except ValueError:
#                 return FloodInputSanitizer(is_valid=False, reason="Input must be a number or 'custom'")
#         except Exception as e:
#             logger.error(f"Input validation error: {str(e)}")
#             return FloodInputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     async def validate_output(self, output: dict) -> FloodOutputSanitizer:
#         """Validate output using simple checks (no API calls)"""
#         try:
#             logger.debug(f"Validating output: {json.dumps(output, indent=2)}")
#             required_keys = ["risk_level", "probability_score", "numbers"]
#             if not all(key in output for key in required_keys):
#                 return FloodOutputSanitizer(is_valid=False, reason="Missing required keys")
#             if output["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
#                 return FloodOutputSanitizer(is_valid=False, reason="Invalid risk_level")
#             if not (0 <= output["probability_score"] <= 1):
#                 return FloodOutputSanitizer(is_valid=False, reason="probability_score out of range (0-1)")
#             return FloodOutputSanitizer(is_valid=True, reason=None)
#         except Exception as e:
#             logger.error(f"Output validation error: {str(e)}")
#             return FloodOutputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

#     async def send_to_cortex(self, prediction_result: dict, alert_message: str, visualization_files: list[str]):
#         """Send processed flood data to Cortex agent"""
#         logger.info("Sending data to Cortex agent...")
#         payload = {
#             "prediction_result": prediction_result,
#             "alert_message": alert_message,
#             "visualization_files": visualization_files
#         }
#         logger.debug(f"Cortex payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
#         print("\n➡️ Sending data to Cortex agent...")
#         print(json.dumps(payload, indent=2, ensure_ascii=False))
#         print("✅ Data sent to Cortex.")

#     async def run(self, input_data: dict) -> HydroMetToolOutput:
#         """Run flood risk analysis"""
#         try:
#             logger.info(f"Received input: {input_data}")
#             # Schema Guardrails
#             hydro_met_input = HydroMetToolInput(**input_data)

#             # Validate input
#             input_str = hydro_met_input.choice
#             if hydro_met_input.choice == "custom":
#                 input_str = ",".join(map(str, hydro_met_input.bbox or []))
#             input_validation = await self.validate_input(input_str)
#             if not input_validation.is_valid:
#                 raise ValueError(f"Invalid input: {input_validation.reason}")

#             # Process input
#             if hydro_met_input.choice == "custom":
#                 if not hydro_met_input.bbox or len(hydro_met_input.bbox) != 4:
#                     raise ValueError("Custom area requires a valid bounding box (min_lat, min_lon, max_lat, max_lon)")
#                 bbox = hydro_met_input.bbox
#                 name = hydro_met_input.area_name or "Custom Area"
#             else:
#                 choice = int(hydro_met_input.choice)
#                 if 1 <= choice <= len(PAKISTAN_AREAS):
#                     area_key = list(PAKISTAN_AREAS.keys())[choice-1]
#                     area_config = PAKISTAN_AREAS[area_key]
#                     bbox = area_config["bbox"]
#                     name = area_config["name"]
#                 else:
#                     raise ValueError(f"Invalid choice: {choice}. Must be 1-{len(PAKISTAN_AREAS)+1}")

#             logger.info(f"Processing flood risk analysis for {name} with bbox {bbox}")

#             # Collect data
#             flood_data = collect_flood_data(bbox, name)
#             logger.debug(f"Flood data collected: {json.dumps(flood_data, indent=2)}")

#             # Run prediction
#             prediction_result = predict_flood_risk(flood_data)
#             logger.debug(f"Prediction result: {json.dumps(prediction_result, indent=2)}")

#             # Validate output
#             output_validation = await self.validate_output(prediction_result)
#             if not output_validation.is_valid:
#                 raise ValueError(f"Invalid output: {output_validation.reason}")

#             # Generate alert message
#             risk_level = prediction_result["risk_level"]
#             urdu_summary = prediction_result["urdu_summary"]
#             recommendations = prediction_result["recommendations"]
#             probability_score = prediction_result["probability_score"]
#             alert_message = (
#                 f"Flood Risk Alert for {name}:\n"
#                 f"Risk Level: {risk_level}\n"
#                 f"Probability Score: {probability_score:.2f}\n"
#                 f"Urdu Summary: {urdu_summary}\n"
#                 f"Recommendations:\n" + "\n".join(recommendations)
#             )
#             if risk_level in ["MEDIUM", "HIGH"]:
#                 alert_message += f"\nTriggering early-warning notifications to downstream AI agents for {risk_level} risk in {name}."

#             # Create Chart.js configuration
#             precip = flood_data["data_sources"]["weather"].get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [0] * 7)
#             days = list(range(1, len(precip) + 1))
#             chart_config = {
#                 "type": "line",
#                 "data": {
#                     "labels": days,
#                     "datasets": [{
#                         "label": "Daily Precipitation (mm)",
#                         "data": precip,
#                         "borderColor": "#1e90ff",
#                         "backgroundColor": "rgba(30, 144, 255, 0.2)",
#                         "fill": True,
#                         "tension": 0.3
#                     }]
#                 },
#                 "options": {
#                     "responsive": True,
#                     "plugins": {
#                         "title": {
#                             "display": True,
#                             "text": f"7-Day Precipitation Forecast for {name} ({flood_data['meta']['datetime']})"
#                         }
#                     },
#                     "scales": {
#                         "x": {
#                             "title": {
#                                 "display": True,
#                                 "text": "Day"
#                             }
#                         },
#                         "y": {
#                             "title": {
#                                 "display": True,
#                                 "text": "Precipitation (mm)"
#                             },
#                             "beginAtZero": True
#                         }
#                     }
#                 }
#             }

#             # Generate visualizations
#             output_dir = 'visualizations'
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             timestamp = flood_data['meta']['datetime'].replace(':', '-')
#             timestamp_display = datetime.strptime(flood_data['meta']['datetime'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
#             precip_days = [
#                 prediction_result['numbers']['forecast_precip_next1_mm'],
#                 prediction_result['numbers']['forecast_precip_next3_mm'] - prediction_result['numbers']['forecast_precip_next1_mm'],
#                 prediction_result['numbers']['forecast_precip_next7_mm'] - prediction_result['numbers']['forecast_precip_next3_mm'],
#                 0, 0, 0, 0
#             ]
#             precip_days = np.array([max(0, x) for x in precip_days[:7]])
#             plt.figure(figsize=(10, 5))
#             plt.plot(range(1, 8), precip_days, marker='o', linestyle='-', color='b')
#             plt.title(f'7-Day Precipitation Forecast for {name} ({timestamp_display})')
#             plt.xlabel('Day')
#             plt.ylabel('Precipitation (mm)')
#             plt.grid(True)
#             plt.tight_layout()
#             precip_filename = f'{output_dir}/precipitation_forecast_{name.replace(" ", "_")}_{timestamp}.png'
#             plt.savefig(precip_filename)
#             plt.close()

#             metrics = ['Runoff (mm)', 'Peak Discharge (m³/s)', 'Time to Peak (hr)', 'AMC (0-1)', 'Anomaly Score (0-1)']
#             values = [
#                 prediction_result['numbers']['estimated_runoff_mm'],
#                 prediction_result['numbers']['estimated_peak_discharge_m3s'],
#                 prediction_result['numbers']['estimated_time_to_peak_hours'],
#                 prediction_result['diagnostics']['amc_info']['amc'],
#                 float(prediction_result['key_factors'][4].split(': ')[1])
#             ]
#             plt.figure(figsize=(12, 6))
#             bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple', 'red'])
#             plt.title(f'Key Flood Forecasting Metrics for {name} ({timestamp_display})')
#             plt.ylabel('Value')
#             plt.grid(axis='y')
#             for bar, value in zip(bars, values):
#                 plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}',
#                          ha='center', va='bottom', fontsize=10)
#             plt.tight_layout()
#             metrics_filename = f'{output_dir}/key_metrics_{name.replace(" ", "_")}_{timestamp}.png'
#             plt.savefig(metrics_filename)
#             plt.close()
#             visualization_files = [precip_filename, metrics_filename]

#             # Store data in SQLite
#             conn = sqlite3.connect('flood_data.db')
#             cursor = conn.cursor()
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS Meta (
#                 id INTEGER PRIMARY KEY,
#                 area_name TEXT,
#                 bbox TEXT,
#                 center TEXT,
#                 timestamp INTEGER,
#                 datetime TEXT,
#                 agent TEXT,
#                 version TEXT,
#                 role TEXT,
#                 completion_time INTEGER
#             )
#             ''')
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS Predictions (
#                 id INTEGER PRIMARY KEY,
#                 meta_id INTEGER,
#                 risk_level TEXT,
#                 probability_score REAL,
#                 forecast_precip_next1_mm REAL,
#                 forecast_precip_next3_mm REAL,
#                 forecast_precip_next7_mm REAL,
#                 curve_number REAL,
#                 estimated_runoff_mm REAL,
#                 estimated_peak_discharge_m3s REAL,
#                 estimated_time_to_peak_hours REAL,
#                 key_factors TEXT,
#                 recommendations TEXT,
#                 urdu_summary TEXT,
#                 diagnostics TEXT,
#                 FOREIGN KEY (meta_id) REFERENCES Meta(id)
#             )
#             ''')
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS DataSources (
#                 id INTEGER PRIMARY KEY,
#                 meta_id INTEGER,
#                 weather TEXT,
#                 elevation TEXT,
#                 water_bodies TEXT,
#                 infrastructure TEXT,
#                 satellite TEXT,
#                 soil TEXT,
#                 landuse TEXT,
#                 river TEXT,
#                 historical TEXT,
#                 satellite_flood TEXT,
#                 FOREIGN KEY (meta_id) REFERENCES Meta(id)
#             )
#             ''')
#             meta = flood_data['meta']
#             cursor.execute('''
#             INSERT INTO Meta (area_name, bbox, center, timestamp, datetime, agent, version, role, completion_time)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta['area_name'],
#                 json.dumps(meta['bbox']),
#                 json.dumps(meta['center']),
#                 meta['timestamp'],
#                 meta['datetime'],
#                 meta['agent'],
#                 meta['version'],
#                 meta['role'],
#                 meta['completion_time']
#             ))
#             meta_id = cursor.lastrowid
#             data_sources = flood_data['data_sources']
#             cursor.execute('''
#             INSERT INTO DataSources (meta_id, weather, elevation, water_bodies, infrastructure, satellite, soil, landuse, river, historical, satellite_flood)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta_id,
#                 json.dumps(data_sources['weather']),
#                 json.dumps(data_sources['elevation']),
#                 json.dumps(data_sources['water_bodies']),
#                 json.dumps(data_sources['infrastructure']),
#                 json.dumps(data_sources['satellite']),
#                 json.dumps(data_sources['soil']),
#                 json.dumps(data_sources['landuse']),
#                 json.dumps(data_sources['river']),
#                 json.dumps(data_sources['historical']),
#                 json.dumps(data_sources['satellite_flood'])
#             ))
#             pred_numbers = prediction_result['numbers']
#             pred_diagnostics = prediction_result['diagnostics']
#             cursor.execute('''
#             INSERT INTO Predictions (meta_id, risk_level, probability_score, forecast_precip_next1_mm, forecast_precip_next3_mm, forecast_precip_next7_mm,
#             curve_number, estimated_runoff_mm, estimated_peak_discharge_m3s, estimated_time_to_peak_hours, key_factors, recommendations, urdu_summary, diagnostics)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 meta_id,
#                 prediction_result['risk_level'],
#                 prediction_result['probability_score'],
#                 pred_numbers['forecast_precip_next1_mm'],
#                 pred_numbers['forecast_precip_next3_mm'],
#                 pred_numbers['forecast_precip_next7_mm'],
#                 pred_numbers['curve_number'],
#                 pred_numbers['estimated_runoff_mm'],
#                 pred_numbers['estimated_peak_discharge_m3s'],
#                 pred_numbers['estimated_time_to_peak_hours'],
#                 json.dumps(prediction_result['key_factors']),
#                 json.dumps(prediction_result['recommendations']),
#                 prediction_result['urdu_summary'],
#                 json.dumps(pred_diagnostics)
#             ))
#             conn.commit()
#             conn.close()
#             logger.info(f"Data stored in SQLite database 'flood_data.db' for {meta['area_name']} at {meta['datetime']}")

#             # Send data to Cortex
#             await self.send_to_cortex(prediction_result, alert_message, visualization_files)

#             return HydroMetToolOutput(
#                 prediction_result=prediction_result,
#                 chart_config=chart_config,
#                 alert_message=alert_message,
#                 visualization_files=visualization_files
#             )

#         except Exception as e:
#             logger.error(f"HydroMetTool error: {str(e)}")
#             raise ValueError(f"HydroMetTool error: {str(e)}")

# hydro_met_tool_instance = HydroMetTool()

# @function_tool
# async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
#     """
#     Tool for Hydro-Meteorological Operations:
#     - Validate input and output
#     - Collect flood data
#     - Predict flood risk
#     - Generate visualizations and alerts
#     """
#     try:
#         logger.debug(f"Tool input: {input_data.model_dump()}")
#         result = await hydro_met_tool_instance.run(input_data.model_dump())
#         logger.debug(f"Tool output before serialization: {result}")
#         serialized_result = result.model_dump()
#         logger.debug(f"Serialized tool output: {serialized_result}")
#         return serialized_result
#     except Exception as e:
#         logger.error(f"Tool execution error: {str(e)}")
#         raise

# # Client and Agent Setup
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# # Initialize the AsyncOpenAI client
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url=BASE_URL
# )

# # Initialize LLM model
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",
#     openai_client=external_client
# )

# # HydroMet Agent
# hydro_met_agent = Agent(
#     name="HydroMetAgent",
#     instructions=(
#         "You are a flood risk forecasting agent for Pakistan. "
#         "Use your LLM to reason about flood scenarios and the hydro_met tool to predict risks and generate alerts."
#     ),
#     model=model,
#     tools=[hydro_met_tool_fn]
# )

# # Main Execution
# if __name__ == "__main__":
#     try:
#         logger.info("Starting standalone HydroMet test")
#         print("\n🌍 Select target area for flood risk analysis:\n")
#         for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
#             print(f"{i}. {val['name']} ({key})")
#         print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")
        
#         choice_input = input("\nEnter choice number: ").strip()
#         tool_input = None
#         if choice_input == str(len(PAKISTAN_AREAS)+1):
#             print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
#             bbox_input = input("Format: lat1,lon1,lat2,lon2 → ").strip()
#             area_name = input("Enter area name (optional, press Enter for 'Custom Area'): ").strip() or "Custom Area"
#             tool_input = HydroMetToolInput(choice="custom", bbox=list(map(float, bbox_input.split(","))), area_name=area_name)
#         else:
#             tool_input = HydroMetToolInput(choice=choice_input)
        
#         logger.debug(f"Tool input before serialization: {tool_input.model_dump()}")
#         serialized_input = json.dumps(tool_input.model_dump())
#         logger.debug(f"Serialized input to Runner: {serialized_input}")
        
#         # Option 1: Test the tool directly to isolate the issue
#         # result = asyncio.run(hydro_met_tool_fn(tool_input))
        
#         # Option 2: Run with agent
#         result = asyncio.run(Runner.run(hydro_met_agent, serialized_input))
#         logger.debug(f"Raw Runner.run output: {result}")
        
#         # Handle different result types
#         if isinstance(result, str):
#             try:
#                 result = json.loads(result)
#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON decode error: {e}")
#                 raise ValueError(f"Failed to parse Runner output: {e}")
#         elif isinstance(result, HydroMetToolOutput):
#             result = result.model_dump()
#         elif hasattr(result, 'final_output'):
#             if isinstance(result.final_output, str):
#                 try:
#                     result = json.loads(result.final_output)
#                 except json.JSONDecodeError as e:
#                     logger.error(f"JSON decode error in final_output: {e}")
#                     raise ValueError(f"Failed to parse Runner final_output: {e}")
#             else:
#                 result = result.final_output
#         else:
#             logger.error(f"Unexpected result type from Runner: {type(result)}")
#             raise ValueError(f"Unexpected result type: {type(result)}")
        
#         print("\n📊 Prediction Results:")
#         print(json.dumps(result["prediction_result"], indent=2, ensure_ascii=False))
        
#         print("\n📈 Precipitation Chart:")
#         print(json.dumps(result["chart_config"], indent=2))
        
#         print("\n🚨 Alerts:")
#         print(result["alert_message"])
        
#         print("\n📊 Visualizations Generated:")
#         for file in result["visualization_files"]:
#             print(f"- {file}")
            
#     except ValueError as e:
#         logger.error(f"Standalone test error: {e}")
#         print(f"Error: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error in standalone test: {e}")
#         print(f"Unexpected error: {e}")




import os
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import logging
try:
    from app.new_agents.hydro_met.data_collection import collect_flood_data, PAKISTAN_AREAS
    from app.new_agents.hydro_met.prediction import predict_flood_risk
except ModuleNotFoundError:
    logging.warning("Failed to import data_collection or prediction. Using mock implementations.")
    PAKISTAN_AREAS = {
        "karachi": {"name": "Karachi Metropolitan", "bbox": [24.8, 66.9, 25.2, 67.3]},
        "lahore": {"name": "Lahore District", "bbox": [31.0, 74.0, 31.5, 74.5]},
        "islamabad": {"name": "Islamabad Capital Territory", "bbox": [33.6, 73.0, 33.8, 73.2]},
        "faisalabad": {"name": "Faisalabad District", "bbox": [31.3, 73.0, 31.5, 73.2]},
        "rawalpindi": {"name": "Rawalpindi District", "bbox": [33.5, 73.0, 33.7, 73.2]},
        "multan": {"name": "Multan District", "bbox": [30.1, 71.4, 30.3, 71.6]},
        "peshawar": {"name": "Peshawar District", "bbox": [34.0, 71.5, 34.2, 71.7]},
        "quetta": {"name": "Quetta District", "bbox": [30.1, 66.9, 30.3, 67.1]}
    }
    def collect_flood_data(bbox, name):
        return {
            "meta": {
                "area_name": name,
                "bbox": bbox,
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "timestamp": int(datetime.now().timestamp()),
                "datetime": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                "agent": "HydroMetAgent",
                "version": "1.0",
                "role": "FloodRiskAnalysis",
                "completion_time": 0
            },
            "data_sources": {
                "weather": {"open_meteo": {"daily": {"precipitation_sum": [10, 20, 30, 40, 50, 60, 70]}}},
                "elevation": {}, "water_bodies": {}, "infrastructure": {}, "satellite": {},
                "soil": {}, "landuse": {}, "river": {}, "historical": {}, "satellite_flood": {}
            }
        }
    def predict_flood_risk(flood_data):
        return {
            "risk_level": "MEDIUM",
            "probability_score": 0.75,
            "urdu_summary": f"{flood_data['meta']['area_name']} میں درمیانے درجے کا سیلاب کا خطرہ",
            "recommendations": ["Monitor river levels", "Prepare evacuation plans"],
            "numbers": {
                "forecast_precip_next1_mm": 10.0,
                "forecast_precip_next3_mm": 25.0,
                "forecast_precip_next7_mm": 50.0,
                "curve_number": 80.0,
                "estimated_runoff_mm": 15.0,
                "estimated_peak_discharge_m3s": 100.0,
                "estimated_time_to_peak_hours": 12.0
            },
            "diagnostics": {"amc_info": {"amc": 0.5}},
            "key_factors": ["Precipitation: High", "Soil Moisture: Moderate", "Elevation: Low", "Land Use: Urban", "Anomaly Score: 0.8"]
        }

from pydantic import BaseModel, field_validator, ConfigDict
from agents import Agent, function_tool, set_tracing_disabled, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
import inspect

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log Runner class details for debugging
logger.debug(f"Runner class: {inspect.getmembers(Runner, predicate=inspect.isfunction)}")

# Load environment variables
load_dotenv()
set_tracing_disabled(True)

# Validation Models
class FloodInputSanitizer(BaseModel):
    is_valid: bool
    reason: str | None = None
    model_config = ConfigDict(extra="forbid")

class FloodOutputSanitizer(BaseModel):
    is_valid: bool
    reason: str | None = None
    model_config = ConfigDict(extra="forbid")

# HydroMetInput and Output Models
class HydroMetToolInput(BaseModel):
    choice: str
    bbox: list[float] | None = None
    area_name: str | None = None
    model_config = ConfigDict(extra="forbid")

    @field_validator('bbox')
    def validate_bbox(cls, v):
        if v and len(v) != 4:
            raise ValueError("Bounding box must contain exactly 4 values: min_lat, min_lon, max_lat, max_lon")
        if v:
            min_lat, min_lon, max_lat, max_lon = v
            if not (0 <= min_lat <= 90 and 0 <= max_lat <= 90 and 0 <= min_lon <= 180 and 0 <= max_lon <= 180):
                raise ValueError("Coordinates out of valid range: latitudes (0-90), longitudes (0-180)")
        return v

class HydroMetToolOutput(BaseModel):
    prediction_result: dict
    chart_config: dict
    alert_message: str
    visualization_files: list[str]
    model_config = ConfigDict(extra="forbid")

# HydroMetTool Class
class HydroMetTool:
    async def validate_input(self, input_str: str) -> FloodInputSanitizer:
        """Validate input using simple checks (no API calls)"""
        try:
            logger.debug(f"Validating input: {input_str}")
            if input_str == "custom":
                return FloodInputSanitizer(is_valid=True, reason="Custom input accepted")
            try:
                choice = int(input_str)
                if 1 <= choice <= len(PAKISTAN_AREAS):
                    return FloodInputSanitizer(is_valid=True, reason=None)
                else:
                    return FloodInputSanitizer(is_valid=False, reason=f"Choice {choice} out of range (1-{len(PAKISTAN_AREAS)})")
            except ValueError:
                # Handle area name input by mapping to index
                area_map = {area["name"].lower(): str(i+1) for i, (key, area) in enumerate(PAKISTAN_AREAS.items())}
                if input_str.lower() in area_map:
                    return FloodInputSanitizer(is_valid=True, reason=f"Mapped area name '{input_str}' to choice '{area_map[input_str.lower()]}'")
                return FloodInputSanitizer(is_valid=False, reason="Input must be a number, 'custom', or a valid area name")
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return FloodInputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

    async def validate_output(self, output: dict) -> FloodOutputSanitizer:
        """Validate output using simple checks (no API calls)"""
        try:
            logger.debug(f"Validating output: {json.dumps(output, indent=2)}")
            required_keys = ["risk_level", "probability_score", "numbers"]
            if not all(key in output for key in required_keys):
                return FloodOutputSanitizer(is_valid=False, reason="Missing required keys")
            if output["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
                return FloodOutputSanitizer(is_valid=False, reason="Invalid risk_level")
            if not (0 <= output["probability_score"] <= 1):
                return FloodOutputSanitizer(is_valid=False, reason="probability_score out of range (0-1)")
            return FloodOutputSanitizer(is_valid=True, reason=None)
        except Exception as e:
            logger.error(f"Output validation error: {str(e)}")
            return FloodOutputSanitizer(is_valid=False, reason=f"Validation error: {str(e)}")

    async def send_to_cortex(self, prediction_result: dict, alert_message: str, visualization_files: list[str]):
        """Send processed flood data to Cortex agent"""
        logger.info("Sending data to Cortex agent...")
        payload = {
            "prediction_result": prediction_result,
            "alert_message": alert_message,
            "visualization_files": visualization_files
        }
        logger.debug(f"Cortex payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        print("\n➡️ Sending data to Cortex agent...")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("✅ Data sent to Cortex.")

    async def run(self, input_data: dict) -> HydroMetToolOutput:
        """Run flood risk analysis"""
        try:
            logger.info(f"Received input: {input_data}")
            # Schema Guardrails
            hydro_met_input = HydroMetToolInput(**input_data)

            # Validate input
            input_str = hydro_met_input.choice
            if hydro_met_input.choice == "custom":
                input_str = ",".join(map(str, hydro_met_input.bbox or []))
            input_validation = await self.validate_input(input_str)
            if not input_validation.is_valid:
                raise ValueError(f"Invalid input: {input_validation.reason}")

            # Process input
            if hydro_met_input.choice == "custom":
                if not hydro_met_input.bbox or len(hydro_met_input.bbox) != 4:
                    raise ValueError("Custom area requires a valid bounding box (min_lat, min_lon, max_lat, max_lon)")
                bbox = hydro_met_input.bbox
                name = hydro_met_input.area_name or "Custom Area"
            else:
                # Handle area name mapping
                area_map = {area["name"].lower(): str(i+1) for i, (key, area) in enumerate(PAKISTAN_AREAS.items())}
                choice = hydro_met_input.choice
                if choice.lower() in area_map:
                    choice = area_map[choice.lower()]
                choice = int(choice)
                if 1 <= choice <= len(PAKISTAN_AREAS):
                    area_key = list(PAKISTAN_AREAS.keys())[choice-1]
                    area_config = PAKISTAN_AREAS[area_key]
                    bbox = area_config["bbox"]
                    name = area_config["name"]
                else:
                    raise ValueError(f"Invalid choice: {choice}. Must be 1-{len(PAKISTAN_AREAS)}")

            logger.info(f"Processing flood risk analysis for {name} with bbox {bbox}")

            # Collect data
            flood_data = collect_flood_data(bbox, name)
            logger.debug(f"Flood data collected: {json.dumps(flood_data, indent=2)}")

            # Run prediction
            prediction_result = predict_flood_risk(flood_data)
            logger.debug(f"Prediction result: {json.dumps(prediction_result, indent=2)}")

            # Validate output
            output_validation = await self.validate_output(prediction_result)
            if not output_validation.is_valid:
                raise ValueError(f"Invalid output: {output_validation.reason}")

            # Generate alert message
            risk_level = prediction_result["risk_level"]
            urdu_summary = prediction_result["urdu_summary"]
            recommendations = prediction_result["recommendations"]
            probability_score = prediction_result["probability_score"]
            alert_message = (
                f"Flood Risk Alert for {name}:\n"
                f"Risk Level: {risk_level}\n"
                f"Probability Score: {probability_score:.2f}\n"
                f"Urdu Summary: {urdu_summary}\n"
                f"Recommendations:\n" + "\n".join(recommendations)
            )
            if risk_level in ["MEDIUM", "HIGH"]:
                alert_message += f"\nTriggering early-warning notifications to downstream AI agents for {risk_level} risk in {name}."

            # Create Chart.js configuration
            precip = flood_data["data_sources"]["weather"].get("open_meteo", {}).get("daily", {}).get("precipitation_sum", [0] * 7)
            days = list(range(1, len(precip) + 1))
            chart_config = {
                "type": "line",
                "data": {
                    "labels": days,
                    "datasets": [{
                        "label": "Daily Precipitation (mm)",
                        "data": precip,
                        "borderColor": "#1e90ff",
                        "backgroundColor": "rgba(30, 144, 255, 0.2)",
                        "fill": True,
                        "tension": 0.3
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"7-Day Precipitation Forecast for {name} ({flood_data['meta']['datetime']})"
                        }
                    },
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Day"
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": "Precipitation (mm)"
                            },
                            "beginAtZero": True
                        }
                    }
                }
            }

            # Generate visualizations
            output_dir = 'visualizations'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = flood_data['meta']['datetime'].replace(':', '-')
            timestamp_display = datetime.strptime(flood_data['meta']['datetime'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            precip_days = [
                prediction_result['numbers']['forecast_precip_next1_mm'],
                prediction_result['numbers']['forecast_precip_next3_mm'] - prediction_result['numbers']['forecast_precip_next1_mm'],
                prediction_result['numbers']['forecast_precip_next7_mm'] - prediction_result['numbers']['forecast_precip_next3_mm'],
                0, 0, 0, 0
            ]
            precip_days = np.array([max(0, x) for x in precip_days[:7]])
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, 8), precip_days, marker='o', linestyle='-', color='b')
            plt.title(f'7-Day Precipitation Forecast for {name} ({timestamp_display})')
            plt.xlabel('Day')
            plt.ylabel('Precipitation (mm)')
            plt.grid(True)
            plt.tight_layout()
            precip_filename = f'{output_dir}/precipitation_forecast_{name.replace(" ", "_")}_{timestamp}.png'
            plt.savefig(precip_filename)
            plt.close()

            metrics = ['Runoff (mm)', 'Peak Discharge (m³/s)', 'Time to Peak (hr)', 'AMC (0-1)', 'Anomaly Score (0-1)']
            values = [
                prediction_result['numbers']['estimated_runoff_mm'],
                prediction_result['numbers']['estimated_peak_discharge_m3s'],
                prediction_result['numbers']['estimated_time_to_peak_hours'],
                prediction_result['diagnostics']['amc_info']['amc'],
                float(prediction_result['key_factors'][4].split(': ')[1])
            ]
            plt.figure(figsize=(12, 6))
            bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple', 'red'])
            plt.title(f'Key Flood Forecasting Metrics for {name} ({timestamp_display})')
            plt.ylabel('Value')
            plt.grid(axis='y')
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}',
                         ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            metrics_filename = f'{output_dir}/key_metrics_{name.replace(" ", "_")}_{timestamp}.png'
            plt.savefig(metrics_filename)
            plt.close()
            visualization_files = [precip_filename, metrics_filename]

            # Store data in SQLite
            conn = sqlite3.connect('flood_data.db')
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Meta (
                id INTEGER PRIMARY KEY,
                area_name TEXT,
                bbox TEXT,
                center TEXT,
                timestamp INTEGER,
                datetime TEXT,
                agent TEXT,
                version TEXT,
                role TEXT,
                completion_time INTEGER
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Predictions (
                id INTEGER PRIMARY KEY,
                meta_id INTEGER,
                risk_level TEXT,
                probability_score REAL,
                forecast_precip_next1_mm REAL,
                forecast_precip_next3_mm REAL,
                forecast_precip_next7_mm REAL,
                curve_number REAL,
                estimated_runoff_mm REAL,
                estimated_peak_discharge_m3s REAL,
                estimated_time_to_peak_hours REAL,
                key_factors TEXT,
                recommendations TEXT,
                urdu_summary TEXT,
                diagnostics TEXT,
                FOREIGN KEY (meta_id) REFERENCES Meta(id)
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS DataSources (
                id INTEGER PRIMARY KEY,
                meta_id INTEGER,
                weather TEXT,
                elevation TEXT,
                water_bodies TEXT,
                infrastructure TEXT,
                satellite TEXT,
                soil TEXT,
                landuse TEXT,
                river TEXT,
                historical TEXT,
                satellite_flood TEXT,
                FOREIGN KEY (meta_id) REFERENCES Meta(id)
            )
            ''')
            meta = flood_data['meta']
            cursor.execute('''
            INSERT INTO Meta (area_name, bbox, center, timestamp, datetime, agent, version, role, completion_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta['area_name'],
                json.dumps(meta['bbox']),
                json.dumps(meta['center']),
                meta['timestamp'],
                meta['datetime'],
                meta['agent'],
                meta['version'],
                meta['role'],
                meta['completion_time']
            ))
            meta_id = cursor.lastrowid
            data_sources = flood_data['data_sources']
            cursor.execute('''
            INSERT INTO DataSources (meta_id, weather, elevation, water_bodies, infrastructure, satellite, soil, landuse, river, historical, satellite_flood)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta_id,
                json.dumps(data_sources['weather']),
                json.dumps(data_sources['elevation']),
                json.dumps(data_sources['water_bodies']),
                json.dumps(data_sources['infrastructure']),
                json.dumps(data_sources['satellite']),
                json.dumps(data_sources['soil']),
                json.dumps(data_sources['landuse']),
                json.dumps(data_sources['river']),
                json.dumps(data_sources['historical']),
                json.dumps(data_sources['satellite_flood'])
            ))
            pred_numbers = prediction_result['numbers']
            pred_diagnostics = prediction_result['diagnostics']
            cursor.execute('''
            INSERT INTO Predictions (meta_id, risk_level, probability_score, forecast_precip_next1_mm, forecast_precip_next3_mm, forecast_precip_next7_mm,
            curve_number, estimated_runoff_mm, estimated_peak_discharge_m3s, estimated_time_to_peak_hours, key_factors, recommendations, urdu_summary, diagnostics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta_id,
                prediction_result['risk_level'],
                prediction_result['probability_score'],
                pred_numbers['forecast_precip_next1_mm'],
                pred_numbers['forecast_precip_next3_mm'],
                pred_numbers['forecast_precip_next7_mm'],
                pred_numbers['curve_number'],
                pred_numbers['estimated_runoff_mm'],
                pred_numbers['estimated_peak_discharge_m3s'],
                pred_numbers['estimated_time_to_peak_hours'],
                json.dumps(prediction_result['key_factors']),
                json.dumps(prediction_result['recommendations']),
                prediction_result['urdu_summary'],
                json.dumps(pred_diagnostics)
            ))
            conn.commit()
            conn.close()
            logger.info(f"Data stored in SQLite database 'flood_data.db' for {meta['area_name']} at {meta['datetime']}")

            # Send data to Cortex
            await self.send_to_cortex(prediction_result, alert_message, visualization_files)

            return HydroMetToolOutput(
                prediction_result=prediction_result,
                chart_config=chart_config,
                alert_message=alert_message,
                visualization_files=visualization_files
            )

        except Exception as e:
            logger.error(f"HydroMetTool error: {str(e)}")
            raise ValueError(f"HydroMetTool error: {str(e)}")

hydro_met_tool_instance = HydroMetTool()

@function_tool
async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
    """
    Tool for Hydro-Meteorological Operations:
    - Validate input and output
    - Collect flood data
    - Predict flood risk
    - Generate visualizations and alerts
    """
    try:
        logger.debug(f"Tool input: {input_data.model_dump()}")
        result = await hydro_met_tool_instance.run(input_data.model_dump())
        logger.debug(f"Tool output before serialization: {result}")
        serialized_result = result.model_dump()
        logger.debug(f"Serialized tool output: {serialized_result}")
        return serialized_result
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        raise

# Client and Agent Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Initialize the AsyncOpenAI client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL
)

# Initialize LLM model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# HydroMet Agent
hydro_met_agent = Agent(
    name="HydroMetAgent",
    instructions=(
        "You are a flood risk forecasting agent for Pakistan. "
        "Use your LLM to reason about flood scenarios and the hydro_met tool to predict risks and generate alerts. "
        "If the input contains an area_name, ensure it is mapped to a valid choice for the hydro_met tool."
    ),
    model=model,
    tools=[hydro_met_tool_fn]
)

# Main Execution
if __name__ == "__main__":
    try:
        logger.info("Starting standalone HydroMet test")
        print("\n🌍 Select target area for flood risk analysis:\n")
        for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
            print(f"{i}. {val['name']} ({key})")
        print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")
        
        choice_input = input("\nEnter choice number: ").strip()
        tool_input = None
        if choice_input == str(len(PAKISTAN_AREAS)+1):
            print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
            bbox_input = input("Format: lat1,lon1,lat2,lon2 → ").strip()
            area_name = input("Enter area name (optional, press Enter for 'Custom Area'): ").strip() or "Custom Area"
            tool_input = HydroMetToolInput(choice="custom", bbox=list(map(float, bbox_input.split(","))), area_name=area_name)
        else:
            tool_input = HydroMetToolInput(choice=choice_input)
        
        logger.debug(f"Tool input before serialization: {tool_input.model_dump()}")
        serialized_input = json.dumps(tool_input.model_dump())
        logger.debug(f"Serialized input to Runner: {serialized_input}")
        
        # Run with agent
        result = asyncio.run(Runner.run(hydro_met_agent, serialized_input))
        logger.debug(f"Raw Runner.run output: {result}")
        
        # Handle different result types
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise ValueError(f"Failed to parse Runner output: {e}")
        elif isinstance(result, HydroMetToolOutput):
            result = result.model_dump()
        elif hasattr(result, 'final_output'):
            if isinstance(result.final_output, str):
                try:
                    result = json.loads(result.final_output)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in final_output: {e}")
                    raise ValueError(f"Failed to parse Runner final_output: {e}")
            else:
                result = result.final_output
        else:
            logger.error(f"Unexpected result type from Runner: {type(result)}")
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        print("\n📊 Prediction Results:")
        print(json.dumps(result["prediction_result"], indent=2, ensure_ascii=False))
        
        print("\n📈 Precipitation Chart:")
        print(json.dumps(result["chart_config"], indent=2))
        
        print("\n🚨 Alerts:")
        print(result["alert_message"])
        
        print("\n📊 Visualizations Generated:")
        for file in result["visualization_files"]:
            print(f"- {file}")
            
    except ValueError as e:
        logger.error(f"Standalone test error: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in standalone test: {e}")
        print(f"Unexpected error: {e}")