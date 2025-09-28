import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Sample data from the output (replace with actual data if needed)
flood_data = {
  "meta": {
    "area_name": "Lahore District",
    "bbox": [31.4, 74.2, 31.7, 74.5],
    "center": [31.55, 74.35],
    "timestamp": 1758984365,
    "datetime": "2025-09-27T19:46:05",
    "agent": "PakistanFloodDataAgent",
    "version": "2.4",
    "role": "Comprehensive flood forecasting data collection and risk assessment",
    "completion_time": 1758984387
  },
  "data_sources": {
    "weather": {
      "open_meteo": {
        "daily": {
          "precipitation_sum": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
      },
      "historical": {
        "daily": {
          "precipitation_sum": [0.0] * 30  # Simplified for brevity
        }
      },
      "nasa": {}
    },
    "elevation": {},
    "water_bodies": {"elements": []},
    "infrastructure": {"elements": []},
    "satellite": {},
    "soil": {
      "soil_moisture": {
        "hourly": {
          "soil_moisture_0_to_7cm": [0.07913333333333332]
        }
      },
      "soil_type": {
        "sand": 40,
        "silt": 30,
        "clay": 25
      }
    },
    "landuse": {},
    "river": {},
    "historical": {},
    "satellite_flood": {}
  },
  "risk_assessment": {},
  "ai_analysis": {},
  "visualization": {},
  "status": "complete"
}

prediction_result = {
  "meta": {
    "generated_at": "2025-09-27T14:46:27.121922Z",
    "area_name": "Lahore District",
    "bbox": [31.4, 74.2, 31.7, 74.5],
    "area_km2": 950.4345522889338
  },
  "risk_level": "LOW",
  "probability_score": 0.2152030319275555,
  "numbers": {
    "forecast_precip_next1_mm": 0.0,
    "forecast_precip_next3_mm": 0.0,
    "forecast_precip_next7_mm": 0.0,
    "curve_number": 65.0,
    "estimated_runoff_mm": 0.0,
    "estimated_peak_discharge_m3s": 0.0,
    "estimated_time_to_peak_hours": 1.608920794127899
  },
  "key_factors": [
    "Antecedent moisture (0-1): 0.08",
    "Next-7-day precipitation (mm): 0.0",
    "Curve Number (CN): 65.0",
    "Waterbody proximity factor (0-1): 1.00",
    "Precipitation anomaly score (0-1): 0.30"
  ],
  "recommendations": [
    "Risk low — continue regular monitoring and keep drainage channels clear.",
    "Share public advisory: low risk but watch for sudden changes in forecast."
  ],
  "urdu_summary": "خلاصہ: موجودہ پیشن گوئی کے مطابق سیلاب کا خطرہ کم ہے۔ موسم کی نگرانی جاری رکھیں۔",
  "diagnostics": {
    "amc_info": {
      "amc": 0.08275555555555555,
      "recent_total_7": 0.3,
      "recent_total_30": 175.60000000000002,
      "soil_moisture_proxy": 0.07913333333333332
    },
    "precip_meta": {
      "f_total": 0.0,
      "h_mean": 6.0551724137931044,
      "h_std": 13.826176727545036,
      "z": -0.4379498781994996
    },
    "rational_meta": {
      "C": 0.1,
      "i_m_per_hr": 0.0,
      "A_m2": 950434552.2889338,
      "tc_hours": 1.608920794127899
    },
    "runoff_components": {
      "runoff_1d_mm": 0.0,
      "runoff_3d_mm": 0.0,
      "runoff_7d_mm": 0.0
    },
    "water_count": 263
  }
}

# Step 1: Store data in SQLite3
def setup_database():
    conn = sqlite3.connect('flood_data.db')
    cursor = conn.cursor()

    # Create tables
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

    return conn, cursor

def store_data(flood_data, prediction_result):
    conn, cursor = setup_database()

    # Insert Meta data
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

    # Insert DataSources
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

    # Insert Predictions
    pred_meta = prediction_result['meta']
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
    print(f"Data stored in SQLite database 'flood_data.db' for {meta['area_name']} at {meta['datetime']}")

# Step 2: Retrieve data from SQLite for visualization
def fetch_data_for_visualization(area_name, timestamp):
    conn = sqlite3.connect('flood_data.db')
    cursor = conn.cursor()

    # Fetch Meta and Predictions data
    cursor.execute('''
    SELECT m.area_name, m.datetime, p.risk_level, p.probability_score, 
           p.forecast_precip_next1_mm, p.forecast_precip_next3_mm, p.forecast_precip_next7_mm,
           p.curve_number, p.estimated_runoff_mm, p.estimated_peak_discharge_m3s,
           p.estimated_time_to_peak_hours, p.key_factors, p.diagnostics
    FROM Meta m
    JOIN Predictions p ON m.id = p.meta_id
    WHERE m.area_name = ? AND m.timestamp = ?
    ''', (area_name, timestamp))
    
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise ValueError(f"No data found for {area_name} at timestamp {timestamp}")

    # Parse result
    area_name, datetime_str, risk_level, probability_score, precip_next1, precip_next3, precip_next7, curve_number, runoff, peak_discharge, time_to_peak, key_factors, diagnostics = result
    key_factors = json.loads(key_factors)
    diagnostics = json.loads(diagnostics)

    return {
        'area_name': area_name,
        'datetime': datetime_str,
        'risk_level': risk_level,
        'probability_score': probability_score,
        'precip_next1': precip_next1,
        'precip_next3': precip_next3,
        'precip_next7': precip_next7,
        'curve_number': curve_number,
        'runoff': runoff,
        'peak_discharge': peak_discharge,
        'time_to_peak': time_to_peak,
        'amc': diagnostics['amc_info']['amc'],
        'anomaly_score': float(key_factors[4].split(': ')[1]),  # Extract from "Precipitation anomaly score (0-1): 0.30"
        'water_count': diagnostics['water_count']
    }

# Step 3: Visualize important features
def visualize_data(data, output_dir='visualizations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    area_name = data['area_name']
    timestamp = data['datetime'].replace(':', '-')  # Replace colons for filename safety
    timestamp_display = datetime.strptime(data['datetime'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

    # Approximate daily precipitation (since data provides cumulative)
    precip_days = [
        data['precip_next1'],
        data['precip_next3'] - data['precip_next1'],
        data['precip_next7'] - data['precip_next3'],
        0, 0, 0, 0
    ]
    precip_days = np.array([max(0, x) for x in precip_days[:7]])  # Ensure non-negative

    # Plot 1: Precipitation Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 8), precip_days, marker='o', linestyle='-', color='b')
    plt.title(f'7-Day Precipitation Forecast for {area_name} ({timestamp_display})')
    plt.xlabel('Day')
    plt.ylabel('Precipitation (mm)')
    plt.grid(True)
    plt.tight_layout()
    precip_filename = f'{output_dir}/precipitation_forecast_{area_name.replace(" ", "_")}_{timestamp}.png'
    plt.savefig(precip_filename)
    plt.close()

    # Plot 2: Key Metrics Bar Chart
    metrics = ['Runoff (mm)', 'Peak Discharge (m³/s)', 'Time to Peak (hr)', 'AMC (0-1)', 'Anomaly Score (0-1)']
    values = [data['runoff'], data['peak_discharge'], data['time_to_peak'], data['amc'], data['anomaly_score']]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple', 'red'])
    plt.title(f'Key Flood Forecasting Metrics for {area_name} ({timestamp_display})')
    plt.ylabel('Value')
    plt.grid(axis='y')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    metrics_filename = f'{output_dir}/key_metrics_{area_name.replace(" ", "_")}_{timestamp}.png'
    plt.savefig(metrics_filename)
    plt.close()

    print(f"Visualizations saved as '{precip_filename}' and '{metrics_filename}'")

# Main execution
if __name__ == "__main__":
    import os
    # Store data
    store_data(flood_data, prediction_result)

    # Fetch data for visualization
    data = fetch_data_for_visualization(
        area_name=flood_data['meta']['area_name'],
        timestamp=flood_data['meta']['timestamp']
    )

    # Generate visualizations
    visualize_data(data)