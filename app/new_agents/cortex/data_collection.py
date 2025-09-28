# data_collection.py
import os
import time
import requests
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np

# -----------------------------
# Load environment & keys
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Cache for API Responses
# -----------------------------
api_cache = {}
CACHE_TTL = 3600

def get_cached_response(cache_key):
    if cache_key in api_cache:
        timestamp, data = api_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_cached_response(cache_key, data):
    api_cache[cache_key] = (time.time(), data)

# -----------------------------
# Helper Functions
# -----------------------------
def safe_get(url, params=None, timeout=15, headers=None, retries=3):
    """Safe HTTP GET with error handling and caching"""
    cache_key = (url, json.dumps(params, sort_keys=True))
    cached = get_cached_response(cache_key)
    if cached:
        return cached
    headers = headers or {'User-Agent': 'FloodForecast/1.0'}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            result = r.json()
            set_cached_response(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"GET failed ({url}, attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def safe_post(url, data=None, timeout=180, headers=None, retries=3):
    """Safe HTTP POST with error handling and caching"""
    cache_key = (url, json.dumps(data, sort_keys=True))
    cached = get_cached_response(cache_key)
    if cached:
        return cached
    headers = headers or {'User-Agent': 'FloodForecast/1.0'}
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, timeout=timeout, headers=headers)
            r.raise_for_status()
            result = r.json()
            set_cached_response(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"POST failed ({url}, attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"elements": []}  # Fallback to empty result

# -----------------------------
# Data Fetch Functions
# -----------------------------
def fetch_weather_data(lat, lon):
    """Comprehensive weather data from multiple sources"""
    weather_data = {}
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "pressure_msl", "cloud_cover"],
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "precipitation_probability_max"],
        "timezone": "Asia/Karachi",
        "forecast_days": 7
    }
    weather_data["open_meteo"] = safe_get(open_meteo_url, params) or {"daily": {"precipitation_sum": [0] * 7}}
    logging.info(f"Open-Meteo forecast response: {json.dumps(weather_data['open_meteo'], indent=2)}")
    historical_url = "https://archive-api.open-meteo.com/v1/archive"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    hist_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
        "timezone": "Asia/Karachi"
    }
    weather_data["historical"] = safe_get(historical_url, hist_params) or {"daily": {"precipitation_sum": [0] * 30}}
    logging.info(f"Open-Meteo historical response: {json.dumps(weather_data['historical'], indent=2)}")
    nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    nasa_params = {
        "parameters": "PRECTOTCORR,T2M,RH2M,WS10M",
        "community": "ag",
        "longitude": lon,
        "latitude": lat,
        "start": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
        "end": datetime.now().strftime("%Y%m%d"),
        "format": "json"
    }
    weather_data["nasa"] = safe_get(nasa_url, nasa_params) or {}
    logging.info(f"NASA weather response: {json.dumps(weather_data['nasa'], indent=2)}")
    return weather_data

def fetch_elevation_data(lat, lon, radius_km=50):
    """Fetch elevation data with retries - Skip USGS for non-US locations"""
    elevation_data = {}
    elevation_data["usgs"] = None
    opentopo_url = "https://api.opentopodata.org/v1/srtm90m"
    elevation_data["opentopo"] = safe_get(opentopo_url, {"locations": f"{lat},{lon}"}) or {}
    open_elev_url = "https://api.open-elevation.com/api/v1/lookup"
    elevation_data["open_elevation"] = safe_get(open_elev_url, {"locations": f"{lat},{lon}"}) or {}
    return elevation_data

def fetch_water_bodies(bbox):
    """Get rivers, lakes, and water bodies from OpenStreetMap"""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:180];
    (
      way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom tags;
    """
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter"
    ]
    for endpoint in overpass_endpoints:
        logging.info(f"Trying Overpass API endpoint: {endpoint}")
        result = safe_post(endpoint, {"data": query})
        if result is not None and "elements" in result:
            logging.info(f"Successfully fetched water bodies from {endpoint}")
            return result
        logging.warning(f"Failed to fetch water bodies from {endpoint}")
    logging.error("All Overpass API endpoints failed; returning empty result")
    return {"elements": []}

def fetch_infrastructure(bbox):
    """Get major roads and critical infrastructure"""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"motorway|trunk|primary"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["amenity"="hospital"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom tags;
    """
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter"
    ]
    for endpoint in overpass_endpoints:
        logging.info(f"Trying Overpass API endpoint: {endpoint}")
        result = safe_post(endpoint, {"data": query})
        if result is not None and "elements" in result:
            logging.info(f"Successfully fetched infrastructure from {endpoint}")
            return result
        logging.warning(f"Failed to fetch infrastructure from {endpoint}")
    logging.error("All Overpass API endpoints failed; returning empty result")
    return {"elements": []}

def fetch_satellite_data(lat, lon):
    """Get satellite-based flood indicators"""
    return {"sentinel_hub": "Register at https://www.sentinel-hub.com/ for Sentinel-2 flood extent data"}

def fetch_soil_data(lat, lon):
    """Fetch soil moisture from Open-Meteo and use static soil type"""
    soil_data = {}
    smap_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "soil_moisture_0_to_7cm",
        "timezone": "Asia/Karachi"
    }
    soil_data["soil_moisture"] = safe_get(smap_url, params) or {"hourly": {"soil_moisture_0_to_7cm": [0.2] * 168}}
    logging.info(f"Soil moisture response: {json.dumps(soil_data['soil_moisture'], indent=2)}")
    soil_data["soil_type"] = {
        "sand": 40,
        "silt": 30,
        "clay": 25,
        "organic_carbon_density": 0.5,
        "note": "Default urban soil profile (sand-heavy, moderate drainage)"
    }
    return soil_data

def fetch_landuse_data(lat, lon, radius_km=5):
    """Fetch land use and runoff data"""
    return {
        "esa_worldcover": "WMS download required (ESA WorldCover)",
        "sentinel2": "Integrate via Google Earth Engine or Sentinel Hub (needs token)"
    }

def fetch_historical_floods(lat, lon):
    """Fetch historical flood data"""
    return {
        "unosat": "Download GeoJSON from https://unosat.org/products/flood",
        "dartmouth": "Check http://floodobservatory.colorado.edu/ for historical flood records"
    }

def fetch_river_data(lat, lon):
    """Fetch river flow/discharge data (placeholder for GloFAS)"""
    return {
        "glofas": {
            "status": "Placeholder - download NetCDF from https://data.jrc.ec.europa.eu/collection/glofas",
            "note": "GloFAS provides river discharge forecasts; implement parsing with xarray for specific rivers"
        }
    }

# -----------------------------
# Flood Risk Indicators
# -----------------------------
def calculate_flood_risk(weather_data, elevation_data, water_bodies):
    """Calculate flood risk based on collected data"""
    risk_factors = {
        "precipitation_risk": 0,
        "elevation_risk": 0,
        "water_proximity_risk": 0,
        "infrastructure_risk": 0,
        "overall_risk": "LOW"
    }
    try:
        if weather_data and "open_meteo" in weather_data:
            forecast = weather_data["open_meteo"]
            if forecast and "daily" in forecast:
                precip = forecast["daily"].get("precipitation_sum", [0])
                precip = [x for x in precip if isinstance(x, (int, float)) and not np.isnan(x)]
                max_precip = max(precip) if precip else 0
                if max_precip > 50:
                    risk_factors["precipitation_risk"] = min(max_precip / 100, 1.0)
        elevation_sources = ["open_elevation", "opentopo"]
        elevation = 500
        for source in elevation_sources:
            if elevation_data and source in elevation_data:
                data = elevation_data[source]
                if data and "results" in data and data["results"]:
                    elevation = data["results"][0].get("elevation", 500)
                    break
                if data and isinstance(data, dict) and "elevation" in data:
                    elevation = data["elevation"]
                    break
        if elevation < 100:
            risk_factors["elevation_risk"] = 0.8
        elif elevation < 300:
            risk_factors["elevation_risk"] = 0.4
        if water_bodies and "elements" in water_bodies:
            water_count = len(water_bodies["elements"])
            risk_factors["water_proximity_risk"] = min(water_count / 10, 1.0)
        avg_risk = np.mean([
            risk_factors["precipitation_risk"],
            risk_factors["elevation_risk"],
            risk_factors["water_proximity_risk"]
        ])
        if avg_risk > 0.7:
            risk_factors["overall_risk"] = "HIGH"
        elif avg_risk > 0.4:
            risk_factors["overall_risk"] = "MEDIUM"
        else:
            risk_factors["overall_risk"] = "LOW"
    except Exception as e:
        logging.error(f"Risk calculation error: {e}")
    return risk_factors

# -----------------------------
# Main Data Collection Orchestrator
# -----------------------------
def collect_flood_data(bbox, target_area_name="Unknown"):
    """Main function to collect all flood-related data"""
    lat = (bbox[0] + bbox[2]) / 2
    lon = (bbox[1] + bbox[3]) / 2
    logging.info(f"🌊 Starting flood data collection for {target_area_name}")
    output = {
        "meta": {
            "area_name": target_area_name,
            "bbox": bbox,
            "center": [lat, lon],
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "agent": "PakistanFloodDataAgent",
            "version": "2.4",
            "role": "Comprehensive flood forecasting data collection and risk assessment"
        },
        "data_sources": {
            "weather": fetch_weather_data(lat, lon),
            "elevation": fetch_elevation_data(lat, lon),
            "water_bodies": fetch_water_bodies(bbox),
            "infrastructure": fetch_infrastructure(bbox),
            "satellite": fetch_satellite_data(lat, lon),
            "soil": fetch_soil_data(lat, lon),
            "landuse": fetch_landuse_data(lat, lon),
            "river": fetch_river_data(lat, lon),
            "historical": fetch_historical_floods(lat, lon),
            "satellite_flood": {"status": "Placeholder - implement Sentinel Hub API"}
        },
        "risk_assessment": {},
        "ai_analysis": {},
        "visualization": {},
        "status": "processing"
    }
    logging.info("📊 Calculating flood risk...")
    output["risk_assessment"] = calculate_flood_risk(
        output["data_sources"]["weather"],
        output["data_sources"]["elevation"],
        output["data_sources"]["water_bodies"]
    )
    if "open_meteo" in output["data_sources"]["weather"]:
        precip = output["data_sources"]["weather"]["open_meteo"].get("daily", {}).get("precipitation_sum", [0] * 7)
        precip = [x for x in precip if isinstance(x, (int, float)) and not np.isnan(x)] or [0] * 7
        days = list(range(1, len(precip) + 1))
        output["visualization"] = {
            "type": "line",
            "data": {
                "labels": days,
                "datasets": [{
                    "label": "Daily Precipitation (mm)",
                    "data": precip,
                    "borderColor": "#1e90ff",
                    "backgroundColor": "rgba(30, 144, 255, 0.2)",
                    "fill": True
                }]
            },
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": "Precipitation (mm)"}},
                    "x": {"title": {"display": True, "text": "Day"}}
                }
            }
        }
    output["status"] = "complete"
    output["meta"]["completion_time"] = int(time.time())
    output_file = f"flood_report_{target_area_name.replace(' ', '_')}_{output['meta']['timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.info(f"📝 Saved flood report to {output_file}")
    logging.info("✅ Flood data collection complete!")
    return output

# -----------------------------
# Pakistan-specific presets
# -----------------------------
PAKISTAN_AREAS = {
    "karachi": {"bbox": [24.7, 66.9, 25.2, 67.5], "name": "Karachi Metropolitan"},
    "lahore": {"bbox": [31.4, 74.2, 31.7, 74.5], "name": "Lahore District"},
    "islamabad": {"bbox": [33.6, 72.8, 33.8, 73.2], "name": "Islamabad Capital Territory"},
    "faisalabad": {"bbox": [31.3, 73.0, 31.5, 73.2], "name": "Faisalabad District"},
    "rawalpindi": {"bbox": [33.5, 73.0, 33.7, 73.2], "name": "Rawalpindi District"},
    "multan": {"bbox": [30.1, 71.4, 30.3, 71.6], "name": "Multan District"},
    "peshawar": {"bbox": [34.0, 71.5, 34.2, 71.7], "name": "Peshawar District"},
    "quetta": {"bbox": [30.1, 66.9, 30.3, 67.1], "name": "Quetta District"}
}# data_collection.py
import os
import time
import requests
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np

# -----------------------------
# Load environment & keys
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Cache for API Responses
# -----------------------------
api_cache = {}
CACHE_TTL = 3600

def get_cached_response(cache_key):
    if cache_key in api_cache:
        timestamp, data = api_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_cached_response(cache_key, data):
    api_cache[cache_key] = (time.time(), data)

# -----------------------------
# Helper Functions
# -----------------------------
def safe_get(url, params=None, timeout=15, headers=None, retries=3):
    """Safe HTTP GET with error handling and caching"""
    cache_key = (url, json.dumps(params, sort_keys=True))
    cached = get_cached_response(cache_key)
    if cached:
        return cached
    headers = headers or {'User-Agent': 'FloodForecast/1.0'}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            r.raise_for_status()
            result = r.json()
            set_cached_response(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"GET failed ({url}, attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def safe_post(url, data=None, timeout=180, headers=None, retries=3):
    """Safe HTTP POST with error handling and caching"""
    cache_key = (url, json.dumps(data, sort_keys=True))
    cached = get_cached_response(cache_key)
    if cached:
        return cached
    headers = headers or {'User-Agent': 'FloodForecast/1.0'}
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, timeout=timeout, headers=headers)
            r.raise_for_status()
            result = r.json()
            set_cached_response(cache_key, result)
            return result
        except Exception as e:
            logging.warning(f"POST failed ({url}, attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"elements": []}  # Fallback to empty result

# -----------------------------
# Data Fetch Functions
# -----------------------------
def fetch_weather_data(lat, lon):
    """Comprehensive weather data from multiple sources"""
    weather_data = {}
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "pressure_msl", "cloud_cover"],
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "precipitation_probability_max"],
        "timezone": "Asia/Karachi",
        "forecast_days": 7
    }
    weather_data["open_meteo"] = safe_get(open_meteo_url, params) or {"daily": {"precipitation_sum": [0] * 7}}
    logging.info(f"Open-Meteo forecast response: {json.dumps(weather_data['open_meteo'], indent=2)}")
    historical_url = "https://archive-api.open-meteo.com/v1/archive"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    hist_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
        "timezone": "Asia/Karachi"
    }
    weather_data["historical"] = safe_get(historical_url, hist_params) or {"daily": {"precipitation_sum": [0] * 30}}
    logging.info(f"Open-Meteo historical response: {json.dumps(weather_data['historical'], indent=2)}")
    nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    nasa_params = {
        "parameters": "PRECTOTCORR,T2M,RH2M,WS10M",
        "community": "ag",
        "longitude": lon,
        "latitude": lat,
        "start": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
        "end": datetime.now().strftime("%Y%m%d"),
        "format": "json"
    }
    weather_data["nasa"] = safe_get(nasa_url, nasa_params) or {}
    logging.info(f"NASA weather response: {json.dumps(weather_data['nasa'], indent=2)}")
    return weather_data

def fetch_elevation_data(lat, lon, radius_km=50):
    """Fetch elevation data with retries - Skip USGS for non-US locations"""
    elevation_data = {}
    elevation_data["usgs"] = None
    opentopo_url = "https://api.opentopodata.org/v1/srtm90m"
    elevation_data["opentopo"] = safe_get(opentopo_url, {"locations": f"{lat},{lon}"}) or {}
    open_elev_url = "https://api.open-elevation.com/api/v1/lookup"
    elevation_data["open_elevation"] = safe_get(open_elev_url, {"locations": f"{lat},{lon}"}) or {}
    return elevation_data

def fetch_water_bodies(bbox):
    """Get rivers, lakes, and water bodies from OpenStreetMap"""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:180];
    (
      way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom tags;
    """
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter"
    ]
    for endpoint in overpass_endpoints:
        logging.info(f"Trying Overpass API endpoint: {endpoint}")
        result = safe_post(endpoint, {"data": query})
        if result is not None and "elements" in result:
            logging.info(f"Successfully fetched water bodies from {endpoint}")
            return result
        logging.warning(f"Failed to fetch water bodies from {endpoint}")
    logging.error("All Overpass API endpoints failed; returning empty result")
    return {"elements": []}

def fetch_infrastructure(bbox):
    """Get major roads and critical infrastructure"""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"motorway|trunk|primary"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["amenity"="hospital"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom tags;
    """
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter"
    ]
    for endpoint in overpass_endpoints:
        logging.info(f"Trying Overpass API endpoint: {endpoint}")
        result = safe_post(endpoint, {"data": query})
        if result is not None and "elements" in result:
            logging.info(f"Successfully fetched infrastructure from {endpoint}")
            return result
        logging.warning(f"Failed to fetch infrastructure from {endpoint}")
    logging.error("All Overpass API endpoints failed; returning empty result")
    return {"elements": []}

def fetch_satellite_data(lat, lon):
    """Get satellite-based flood indicators"""
    return {"sentinel_hub": "Register at https://www.sentinel-hub.com/ for Sentinel-2 flood extent data"}

def fetch_soil_data(lat, lon):
    """Fetch soil moisture from Open-Meteo and use static soil type"""
    soil_data = {}
    smap_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "soil_moisture_0_to_7cm",
        "timezone": "Asia/Karachi"
    }
    soil_data["soil_moisture"] = safe_get(smap_url, params) or {"hourly": {"soil_moisture_0_to_7cm": [0.2] * 168}}
    logging.info(f"Soil moisture response: {json.dumps(soil_data['soil_moisture'], indent=2)}")
    soil_data["soil_type"] = {
        "sand": 40,
        "silt": 30,
        "clay": 25,
        "organic_carbon_density": 0.5,
        "note": "Default urban soil profile (sand-heavy, moderate drainage)"
    }
    return soil_data

def fetch_landuse_data(lat, lon, radius_km=5):
    """Fetch land use and runoff data"""
    return {
        "esa_worldcover": "WMS download required (ESA WorldCover)",
        "sentinel2": "Integrate via Google Earth Engine or Sentinel Hub (needs token)"
    }

def fetch_historical_floods(lat, lon):
    """Fetch historical flood data"""
    return {
        "unosat": "Download GeoJSON from https://unosat.org/products/flood",
        "dartmouth": "Check http://floodobservatory.colorado.edu/ for historical flood records"
    }

def fetch_river_data(lat, lon):
    """Fetch river flow/discharge data (placeholder for GloFAS)"""
    return {
        "glofas": {
            "status": "Placeholder - download NetCDF from https://data.jrc.ec.europa.eu/collection/glofas",
            "note": "GloFAS provides river discharge forecasts; implement parsing with xarray for specific rivers"
        }
    }

# -----------------------------
# Flood Risk Indicators
# -----------------------------
def calculate_flood_risk(weather_data, elevation_data, water_bodies):
    """Calculate flood risk based on collected data"""
    risk_factors = {
        "precipitation_risk": 0,
        "elevation_risk": 0,
        "water_proximity_risk": 0,
        "infrastructure_risk": 0,
        "overall_risk": "LOW"
    }
    try:
        if weather_data and "open_meteo" in weather_data:
            forecast = weather_data["open_meteo"]
            if forecast and "daily" in forecast:
                precip = forecast["daily"].get("precipitation_sum", [0])
                precip = [x for x in precip if isinstance(x, (int, float)) and not np.isnan(x)]
                max_precip = max(precip) if precip else 0
                if max_precip > 50:
                    risk_factors["precipitation_risk"] = min(max_precip / 100, 1.0)
        elevation_sources = ["open_elevation", "opentopo"]
        elevation = 500
        for source in elevation_sources:
            if elevation_data and source in elevation_data:
                data = elevation_data[source]
                if data and "results" in data and data["results"]:
                    elevation = data["results"][0].get("elevation", 500)
                    break
                if data and isinstance(data, dict) and "elevation" in data:
                    elevation = data["elevation"]
                    break
        if elevation < 100:
            risk_factors["elevation_risk"] = 0.8
        elif elevation < 300:
            risk_factors["elevation_risk"] = 0.4
        if water_bodies and "elements" in water_bodies:
            water_count = len(water_bodies["elements"])
            risk_factors["water_proximity_risk"] = min(water_count / 10, 1.0)
        avg_risk = np.mean([
            risk_factors["precipitation_risk"],
            risk_factors["elevation_risk"],
            risk_factors["water_proximity_risk"]
        ])
        if avg_risk > 0.7:
            risk_factors["overall_risk"] = "HIGH"
        elif avg_risk > 0.4:
            risk_factors["overall_risk"] = "MEDIUM"
        else:
            risk_factors["overall_risk"] = "LOW"
    except Exception as e:
        logging.error(f"Risk calculation error: {e}")
    return risk_factors

# -----------------------------
# Main Data Collection Orchestrator
# -----------------------------
def collect_flood_data(bbox, target_area_name="Unknown"):
    """Main function to collect all flood-related data"""
    lat = (bbox[0] + bbox[2]) / 2
    lon = (bbox[1] + bbox[3]) / 2
    logging.info(f"🌊 Starting flood data collection for {target_area_name}")
    output = {
        "meta": {
            "area_name": target_area_name,
            "bbox": bbox,
            "center": [lat, lon],
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "agent": "PakistanFloodDataAgent",
            "version": "2.4",
            "role": "Comprehensive flood forecasting data collection and risk assessment"
        },
        "data_sources": {
            "weather": fetch_weather_data(lat, lon),
            "elevation": fetch_elevation_data(lat, lon),
            "water_bodies": fetch_water_bodies(bbox),
            "infrastructure": fetch_infrastructure(bbox),
            "satellite": fetch_satellite_data(lat, lon),
            "soil": fetch_soil_data(lat, lon),
            "landuse": fetch_landuse_data(lat, lon),
            "river": fetch_river_data(lat, lon),
            "historical": fetch_historical_floods(lat, lon),
            "satellite_flood": {"status": "Placeholder - implement Sentinel Hub API"}
        },
        "risk_assessment": {},
        "ai_analysis": {},
        "visualization": {},
        "status": "processing"
    }
    logging.info("📊 Calculating flood risk...")
    output["risk_assessment"] = calculate_flood_risk(
        output["data_sources"]["weather"],
        output["data_sources"]["elevation"],
        output["data_sources"]["water_bodies"]
    )
    if "open_meteo" in output["data_sources"]["weather"]:
        precip = output["data_sources"]["weather"]["open_meteo"].get("daily", {}).get("precipitation_sum", [0] * 7)
        precip = [x for x in precip if isinstance(x, (int, float)) and not np.isnan(x)] or [0] * 7
        days = list(range(1, len(precip) + 1))
        output["visualization"] = {
            "type": "line",
            "data": {
                "labels": days,
                "datasets": [{
                    "label": "Daily Precipitation (mm)",
                    "data": precip,
                    "borderColor": "#1e90ff",
                    "backgroundColor": "rgba(30, 144, 255, 0.2)",
                    "fill": True
                }]
            },
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": "Precipitation (mm)"}},
                    "x": {"title": {"display": True, "text": "Day"}}
                }
            }
        }
    output["status"] = "complete"
    output["meta"]["completion_time"] = int(time.time())
    output_file = f"flood_report_{target_area_name.replace(' ', '_')}_{output['meta']['timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.info(f"📝 Saved flood report to {output_file}")
    logging.info("✅ Flood data collection complete!")
    return output

# -----------------------------
# Pakistan-specific presets
# -----------------------------
PAKISTAN_AREAS = {
    "karachi": {"bbox": [24.7, 66.9, 25.2, 67.5], "name": "Karachi Metropolitan"},
    "lahore": {"bbox": [31.4, 74.2, 31.7, 74.5], "name": "Lahore District"},
    "islamabad": {"bbox": [33.6, 72.8, 33.8, 73.2], "name": "Islamabad Capital Territory"},
    "faisalabad": {"bbox": [31.3, 73.0, 31.5, 73.2], "name": "Faisalabad District"},
    "rawalpindi": {"bbox": [33.5, 73.0, 33.7, 73.2], "name": "Rawalpindi District"},
    "multan": {"bbox": [30.1, 71.4, 30.3, 71.6], "name": "Multan District"},
    "peshawar": {"bbox": [34.0, 71.5, 34.2, 71.7], "name": "Peshawar District"},
    "quetta": {"bbox": [30.1, 66.9, 30.3, 67.1], "name": "Quetta District"}
}