# import os
# import json
# import sqlite3
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
# from dotenv import load_dotenv
# from data_collection import collect_flood_data, PAKISTAN_AREAS
# from prediction import predict_flood_risk
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

# from pydantic import BaseModel, field_validator, ConfigDict

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

# from agents import function_tool
# import asyncio

# hydro_met_tool_instance = HydroMetTool()

# @function_tool
# async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
#     """Agent-ready HydroMet tool"""
#     result = await hydro_met_tool_instance.run(input_data.model_dump())
#     return result.model_dump()



# from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
# import os

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client)

# hydro_met_agent = Agent(
#     name="HydroMetAgent",
#     instructions=(
#         "You are a flood risk forecasting agent for Pakistan. "
#         "Use your LLM to reason about flood scenarios and call the hydro_met tool "
#         "to predict risks, generate alerts, and create visualizations. "
#         "Ensure the input is a valid choice number, area name, or custom bounding box."
#     ),
#     model=model,
#     tools=[hydro_met_tool_fn],
# )
# # from agents import Agent, OpenAIChatCompletionsModel, Runner

# # agent = Agent(
# #     name="HydroMetTestAgent",
# #     instructions="Just call the hydro_met tool",
# #     model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=client),
# #     tools=[hydro_met_tool_fn]
# # )

# # async def main():
# #     tool_input = {
# #         "choice": "custom",
# #         "bbox": [24.0, 67.0, 25.0, 68.0],
# #         "area_name": "Test Area"
# #     }

# #     result = await Runner.run(agent, [{"role": "user", "content": HydroMetToolInput(**tool_input)}])
# #     print("Tool result:", result.final_output)
# # if __name__ == "__main__":
# #     import asyncio
# #     asyncio.run(main())



# # import json
# # from typing_extensions import TypedDict, Any
# # from agents import Agent, FunctionTool, RunContextWrapper, function_tool
# # from pydantic import BaseModel

# # # ---------------------------
# # # Input model for HydroMet tool
# # # ---------------------------
# # class HydroMetToolInput(BaseModel):
# #     choice: str                 # Predefined area number ("1"-"n") or "custom"
# #     bbox: list[float] | None = None  # Required if choice="custom"; [min_lat, min_lon, max_lat, max_lon]
# #     area_name: str | None = None      # Optional name for custom area

# # # ---------------------------
# # # FunctionTool definition
# # # ---------------------------
# # @function_tool
# # async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
# #     """
# #     Run HydroMet flood risk analysis.

# #     Args:
# #         input_data (HydroMetToolInput): Contains the choice, bounding box (if custom), and area_name.

# #     Choice parameter can be either:
# #       - A number corresponding to a predefined area in Pakistan:
# #           1 → Karachi
# #           2 → Lahore
# #           3 → Islamabad
# #           4 → Peshawar
# #           5 → Quetta
# #           6 → Multan
# #           7 → Faisalabad
# #           8 → Hyderabad
# #           9 → Sialkot
# #       - The string "custom" to provide a bounding box with format [min_lat, min_lon, max_lat, max_lon] and an optional area_name.

# #     Returns:
# #         dict: HydroMetOutput containing:
# #             - prediction_result: dict with risk_level, probability_score, and numerical metrics
# #             - chart_config: dict containing Chart.js configuration for precipitation
# #             - alert_message: str with risk summary and recommendations
# #             - visualization_files: list of file paths to generated charts
# #     """
# #     result = await hydro_met_tool_instance.run(input_data.model_dump())
# #     return result.model_dump()

# # # ---------------------------
# # # Agent setup
# # # ---------------------------
# # agent = Agent(
# #     name="HydroMetAgent",
# #     tools=[hydro_met_tool_fn],
# # )

# # # ---------------------------
# # # Print all tools & their parameters
# # # ---------------------------
# # for tool in agent.tools:
# #     if isinstance(tool, FunctionTool):
# #         print(tool.name)
# #         print(tool.description)
# #         print(json.dumps(tool.params_json_schema, indent=2))
# #         print()
# # import asyncio
# # import json
# # from agents import Runner, ToolCallOutputItem

# # # ---------------------------
# # # Custom output extractor
# # # ---------------------------
# # async def extract_json_payload(run_result) -> str:
# #     """
# #     Extract only the JSON payload from the tool output.
# #     Scans the outputs in reverse until it finds a JSON-like string.
# #     """
# #     for item in reversed(run_result.new_items):
# #         if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
# #             return item.output.strip()
# #     return "{}"  # fallback if nothing found

# # # ---------------------------
# # # Example runner
# # # ---------------------------
# # async def main():
# #     # Example input: custom area
# #     tool_input = {
# #         "choice": "custom",
# #         "bbox": [24.0, 67.0, 25.0, 68.0],
# #         "area_name": "Lahore"
# #     }

# #     # Run the agent with the tool input
# #     result = await Runner.run(agent, [{"role": "user", "content": json.dumps(tool_input)}])

# #     # Extract JSON payload
# #     json_output = await extract_json_payload(result)

# #     # Pretty-print the output
# #     print("HydroMet JSON output:\n", json.dumps(json.loads(json_output), indent=2, ensure_ascii=False))

# # # ---------------------------
# # # Run
# # # ---------------------------
# # if __name__ == "__main__":
# #     asyncio.run(main())


# # 📦 Import Required Libraries
# import os
# import json
# from dotenv import load_dotenv
# from agents import (
#     Agent,
#     Runner,
#     AsyncOpenAI,
#     OpenAIChatCompletionsModel,
#     function_tool,
#     set_tracing_disabled
# )
# from pydantic import BaseModel

# # 🌿 Load environment variables
# load_dotenv()
# set_tracing_disabled(disabled=True)  # Optional: disable internal tracing

# # 🔐 1) Gemini client setup
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# gemini_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url=BASE_URL
# )

# # 🧠 2) Initialize model with Gemini client
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",
#     openai_client=gemini_client
# )

# # ---------------------------
# # 3) HydroMet tool input/output models
# # ---------------------------
# class HydroMetToolInput(BaseModel):
#     choice: str                     # 1-9 or "custom"
#     bbox: list[float] | None = None
#     area_name: str | None = None

# class HydroMetToolOutput(BaseModel):
#     prediction_result: dict
#     chart_config: dict
#     alert_message: str
#     visualization_files: list[str]

# # ---------------------------
# # 4) HydroMet tool function
# # ---------------------------
# @function_tool
# async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
#     """
#     Simulated HydroMet flood prediction.
#     Replace `simulate_prediction` with your actual HydroMet logic.
#     """
#     # --- Input processing ---
#     if input_data.choice == "custom":
#         area_name = input_data.area_name or "Custom Area"
#         bbox = input_data.bbox or [0,0,1,1]
#     else:
#         predefined_areas = {
#             "1": "Karachi",
#             "2": "Lahore",
#             "3": "Islamabad",
#             "4": "Peshawar",
#             "5": "Quetta",
#             "6": "Multan",
#             "7": "Faisalabad",
#             "8": "Hyderabad",
#             "9": "Sialkot"
#         }
#         area_name = predefined_areas.get(input_data.choice, "Unknown Area")
#         bbox = [0,0,1,1]  # Dummy bbox

#     # --- Simulate prediction (replace with real HydroMet call) ---
#     prediction_result = {
#         "risk_level": "MEDIUM",
#         "probability_score": 0.63,
#         "numbers": {
#             "forecast_precip_next1_mm": 12,
#             "forecast_precip_next3_mm": 30,
#             "forecast_precip_next7_mm": 55,
#             "estimated_runoff_mm": 5,
#             "estimated_peak_discharge_m3s": 2.2,
#             "estimated_time_to_peak_hours": 3.5,
#             "curve_number": 75
#         },
#         "diagnostics": {
#             "amc_info": {"amc": 0.8},
#             "water_count": 3
#         },
#         "key_factors": ["Factor1: 1", "Factor2: 2", "Anomaly Score: 0.4"],
#         "recommendations": ["Check drainage", "Prepare alerts"],
#         "urdu_summary": "متوسط سیلاب کا خطرہ"
#     }

#     chart_config = {
#         "type": "line",
#         "data": {
#             "labels": [1,2,3,4,5,6,7],
#             "datasets": [{"label":"Daily Precipitation","data":[12,9,14,8,10,7,5]}]
#         }
#     }

#     alert_message = (
#         f"Flood Risk Alert for {area_name}:\n"
#         f"Risk Level: {prediction_result['risk_level']}\n"
#         f"Probability Score: {prediction_result['probability_score']}\n"
#         f"Urdu Summary: {prediction_result['urdu_summary']}\n"
#         f"Recommendations: {', '.join(prediction_result['recommendations'])}"
#     )

#     visualization_files = ["precipitation.png", "metrics.png"]

#     return HydroMetToolOutput(
#         prediction_result=prediction_result,
#         chart_config=chart_config,
#         alert_message=alert_message,
#         visualization_files=visualization_files
#     ).model_dump()

# # ---------------------------
# # 5) Agent setup
# # ---------------------------
# hydro_met_agent = Agent(
#     name="HydroMetAgent",
#     instructions=(
#         "You are a flood risk forecasting assistant for Pakistan. "
#         "Use the HydroMet tool to generate flood predictions and alerts. "
#         "Ensure input is valid (1-9 for predefined cities or 'custom' bbox)."
#     ),
#     model=model,
#     tools=[hydro_met_tool_fn]
# )

# # ---------------------------
# # 6) Run agent with example input
# # ---------------------------
# async def main():
#     example_input = {
#         "choice": "custom",
#         "bbox": [24.0, 67.0, 25.0, 68.0],
#         "area_name": "Test Area"
#     }
#     result = await Runner.run(hydro_met_agent, [{"role": "user", "content": json.dumps(example_input)}])
#     print("\n✅ Agent Output:\n")
#     print(result.final_output)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())



# import os
# import json
# import sqlite3
# from datetime import datetime
# from dotenv import load_dotenv
# import matplotlib.pyplot as plt
# import numpy as np

# from flask import Flask, request, jsonify, send_from_directory
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
# from pydantic import BaseModel

# # ---------------------------
# # 1️⃣ Environment
# # ---------------------------
# load_dotenv()
# set_tracing_disabled(disabled=True)

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# gemini_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
# model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)

# # ---------------------------
# # 2️⃣ Input/Output Models
# # ---------------------------
# class HydroMetToolInput(BaseModel):
#     choice: str
#     bbox: list[float] | None = None
#     area_name: str | None = None

# class HydroMetToolOutput(BaseModel):
#     prediction_result: dict
#     chart_config: dict
#     alert_message: str
#     visualization_files: list[str]

# # ---------------------------
# # 3️⃣ HydroMet tool function
# # ---------------------------
# @function_tool
# async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
#     # Process input
#     if input_data.choice == "custom":
#         area_name = input_data.area_name or "Custom Area"
#         bbox = input_data.bbox or [0,0,1,1]
#     else:
#         predefined_areas = {
#             "1":"Karachi","2":"Lahore","3":"Islamabad","4":"Peshawar",
#             "5":"Quetta","6":"Multan","7":"Faisalabad","8":"Hyderabad","9":"Sialkot"
#         }
#         area_name = predefined_areas.get(input_data.choice,"Unknown Area")
#         bbox = [0,0,1,1]

#     # Simulated prediction
#     prediction_result = {
#         "risk_level":"MEDIUM",
#         "probability_score":0.63,
#         "numbers":{
#             "forecast_precip_next1_mm":12,
#             "forecast_precip_next3_mm":30,
#             "forecast_precip_next7_mm":55,
#             "estimated_runoff_mm":5,
#             "estimated_peak_discharge_m3s":2.2,
#             "estimated_time_to_peak_hours":3.5,
#             "curve_number":75
#         },
#         "diagnostics":{"amc_info":{"amc":0.8},"water_count":3},
#         "key_factors":["Factor1:1","Factor2:2","Anomaly Score:0.4"],
#         "recommendations":["Check drainage","Prepare alerts"],
#         "urdu_summary":"متوسط سیلاب کا خطرہ"
#     }

#     # Generate charts
#     output_dir = "visualizations"
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

#     # Precipitation chart
#     precip_days = [12,9,14,8,10,7,5]
#     plt.figure(figsize=(8,4))
#     plt.plot(range(1,8), precip_days, marker='o', linestyle='-', color='b')
#     plt.title(f"7-Day Precipitation Forecast for {area_name}")
#     plt.xlabel("Day"); plt.ylabel("Precipitation (mm)")
#     plt.grid(True); plt.tight_layout()
#     precip_filename = f"{output_dir}/precip_{area_name.replace(' ','_')}_{timestamp}.png"
#     plt.savefig(precip_filename); plt.close()

#     # Metrics chart
#     metrics = ['Runoff','Peak Discharge','Time to Peak','AMC','Anomaly Score']
#     values = [5,2.2,3.5,0.8,0.4]
#     plt.figure(figsize=(8,4))
#     bars = plt.bar(metrics, values, color=['blue','green','orange','purple','red'])
#     plt.title(f"Key Metrics for {area_name}")
#     plt.ylabel("Value"); plt.grid(axis='y')
#     for bar, val in zip(bars, values):
#         plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha='center', va='bottom')
#     plt.tight_layout()
#     metrics_filename = f"{output_dir}/metrics_{area_name.replace(' ','_')}_{timestamp}.png"
#     plt.savefig(metrics_filename); plt.close()

#     visualization_files = [precip_filename, metrics_filename]

#     # Store in SQLite
#     conn = sqlite3.connect('flood_data.db'); c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS Predictions (
#         id INTEGER PRIMARY KEY, area_name TEXT, risk_level TEXT, probability_score REAL,
#         timestamp TEXT, visualization_files TEXT
#     )''')
#     c.execute("INSERT INTO Predictions (area_name,risk_level,probability_score,timestamp,visualization_files) VALUES (?,?,?,?,?)",
#               (area_name,prediction_result["risk_level"],prediction_result["probability_score"],timestamp,json.dumps(visualization_files)))
#     conn.commit(); conn.close()

#     alert_message = f"Flood Risk Alert for {area_name}: {prediction_result['risk_level']} risk. Urdu: {prediction_result['urdu_summary']}"
#     chart_config = {"type":"line","data":{"labels":list(range(1,8)),"datasets":[{"label":"Daily Precipitation","data":precip_days}]}}
    
#     return HydroMetToolOutput(prediction_result=prediction_result,
#                               chart_config=chart_config,
#                               alert_message=alert_message,
#                               visualization_files=visualization_files).model_dump()

# # ---------------------------
# # 4️⃣ Agent
# # ---------------------------
# hydro_met_agent = Agent(
#     name="HydroMetAgent",
#     instructions="Use the HydroMet tool to generate flood predictions and alerts.",
#     model=model,
#     tools=[hydro_met_tool_fn]
# )

# # ---------------------------
# # 5️⃣ Flask Web Server
# # ---------------------------
# app = Flask(__name__, static_folder="visualizations")

# @app.route("/flood_forecast", methods=["POST"])
# async def flood_forecast():
#     data = await request.get_json(force=True)
#     choice = data.get("choice","1")
#     bbox = data.get("bbox", None)
#     area_name = data.get("area_name", None)
#     result = await Runner.run(hydro_met_agent, [{"role":"user","content":json.dumps({
#         "choice": choice, "bbox": bbox, "area_name": area_name
#     })}])
#     return jsonify(result.final_output)

# @app.route("/visualizations/<path:filename>")
# def serve_visualization(filename):
#     return send_from_directory("visualizations", filename)
# @app.route("/")
# def index():
#     return """
#     <h2>HydroMet Flood Prediction API</h2>
#     <p>Use <code>/flood_forecast</code> POST to get predictions.</p>
#     <p>Generated charts are in <code>/visualizations/&lt;filename&gt;</code></p>
#     """

# import nest_asyncio
# if __name__ == "__main__":
#     import asyncio
    
#     nest_asyncio.apply()  # Allows Flask + asyncio in same thread
#     app.run(host="0.0.0.0", port=5000, debug=True)


from data_collection import collect_flood_data, PAKISTAN_AREAS
from prediction import predict_flood_risk

import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from pydantic import BaseModel, field_validator

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
# 2️⃣ Input/Output Models
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
# 3️⃣ HydroMet Tool Function
# ---------------------------
@function_tool
async def hydro_met_tool_fn(input_data: HydroMetToolInput) -> dict:
    """
    Generate flood predictions and visualizations for a specified area in Pakistan.

    Args:
        input_data (HydroMetToolInput):
            - choice: string, either:
                * "custom" → for user-defined areas
                * predefined area number or key (from the list below):
                  1: Karachi
                  2: Lahore
                  3: Islamabad
                  4: Peshawar
                  5: Quetta
                  6: Multan
                  7: Faisalabad
                  8: Hyderabad
                  9: Sialkot
            - bbox: list of four floats [min_lat, min_lon, max_lat, max_lon]
                    Required if choice="custom"
            - area_name: optional string, only used if choice="custom"

    Returns:
        HydroMetToolOutput dict containing:
            - prediction_result: dictionary with flood prediction metrics
            - chart_config: chart info for plotting
            - alert_message: formatted alert message
            - visualization_files: list of generated image file paths
    """

    # --- Determine area ---
    if input_data.choice == "custom":
        area_name = input_data.area_name or "Custom Area"
        bbox = input_data.bbox or [0, 0, 1, 1]
    else:
        if input_data.choice not in PAKISTAN_AREAS:
            predefined_keys = list(PAKISTAN_AREAS.keys())
            idx = int(input_data.choice) - 1
            if idx < 0 or idx >= len(predefined_keys):
                raise ValueError("Invalid choice number")
            area_key = predefined_keys[idx]
            area_config = PAKISTAN_AREAS[area_key]
            bbox = area_config["bbox"]
            area_name = area_config["name"]
        else:
            area_config = PAKISTAN_AREAS[input_data.choice]
            bbox = area_config["bbox"]
            area_name = area_config["name"]

    # --- Collect real data ---
    flood_data = collect_flood_data(bbox, area_name)

    # --- Run real prediction ---
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
        float(prediction_result['key_factors'][2].split(':')[1])  # Anomaly score
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
# 4️⃣ Agent
# ---------------------------
hydro_met_agent = Agent(
    name="HydroMetAgent",
    instructions=(
        "Use the HydroMet tool to generate flood predictions and alerts.\n\n"
        "Pass a JSON object with keys:\n"
        "  - choice: 'custom' or one of the predefined numbers/keys:\n"
        "      1: Karachi\n"
        "      2: Lahore\n"
        "      3: Islamabad\n"
        "      4: Peshawar\n"
        "      5: Quetta\n"
        "      6: Multan\n"
        "      7: Faisalabad\n"
        "      8: Hyderabad\n"
        "      9: Sialkot\n"
        "  - bbox: list of four floats [min_lat, min_lon, max_lat, max_lon], required if choice='custom'\n"
        "  - area_name: optional string, only needed if choice='custom'\n\n"
        "The tool returns prediction metrics, charts, an alert message, and visualization file paths."
    ),
    model=model,
    tools=[hydro_met_tool_fn]
)


# ---------------------------
# 5️⃣ Flask Web Server
# ---------------------------
app = Flask(__name__, static_folder="visualizations")

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

import os
from flask import send_file

@app.route("/visualizations")
def list_visualizations():
    files = os.listdir("visualizations")
    links = [f'<a href="/visualizations/{f}">{f}</a>' for f in files]
    return "<br>".join(links)

@app.route("/")
def index():
    return """
    <h2>HydroMet Flood Prediction API</h2>
    <p>Use <code>/flood_forecast</code> POST to get predictions.</p>
    <p>Generated charts are in <code>/visualizations/&lt;filename&gt;</code></p>
    """

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    app.run(host="0.0.0.0", port=5000, debug=True)
