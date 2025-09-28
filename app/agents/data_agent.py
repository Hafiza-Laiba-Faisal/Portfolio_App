
# # import os
# # import time
# # import requests
# # import logging
# # import json
# # from datetime import datetime, timedelta
# # from dotenv import load_dotenv
# # from google import genai
# # import numpy as np

# # # -----------------------------
# # # 1️⃣ Load environment & keys
# # # -----------------------------
# # load_dotenv()
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # if not GEMINI_API_KEY:
# #     raise RuntimeError("❌ GEMINI_API_KEY not found. Check your .env file!")

# # client = genai.Client(api_key=GEMINI_API_KEY)

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # -----------------------------
# # # 2️⃣ Helper Functions
# # # -----------------------------
# # def safe_get(url, params=None, timeout=15, headers=None):
# #     """Safe HTTP GET with error handling"""
# #     try:
# #         headers = headers or {'User-Agent': 'FloodForecast/1.0'}
# #         r = requests.get(url, params=params, timeout=timeout, headers=headers)
# #         r.raise_for_status()
# #         return r.json()
# #     except Exception as e:
# #         logging.warning(f"GET failed ({url}): {e}")
# #         return None

# # def safe_post(url, data=None, timeout=15, headers=None):
# #     """Safe HTTP POST with error handling"""
# #     try:
# #         headers = headers or {'User-Agent': 'FloodForecast/1.0'}
# #         r = requests.post(url, data=data, timeout=timeout, headers=headers)
# #         r.raise_for_status()
# #         return r.json()
# #     except Exception as e:
# #         logging.warning(f"POST failed ({url}): {e}")
# #         return None

# # # -----------------------------
# # # 3️⃣ Enhanced Data Sources
# # # -----------------------------

# # # 🌦️ Weather & Rainfall (Multiple sources)
# # def fetch_weather_data(lat, lon):
# #     """Comprehensive weather data from multiple sources"""
# #     weather_data = {}
    
# #     # Open-Meteo (Primary - very reliable)
# #     open_meteo_url = "https://api.open-meteo.com/v1/forecast"
# #     params = {
# #         "latitude": lat,
# #         "longitude": lon,
# #         "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", 
# #                   "wind_speed_10m", "pressure_msl", "cloud_cover"],
# #         "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
# #                  "wind_speed_10m_max", "precipitation_probability_max"],
# #         "timezone": "Asia/Karachi",
# #         "forecast_days": 7
# #     }
# #     weather_data["open_meteo"] = safe_get(open_meteo_url, params)
    
# #     # Historical weather for context
# #     historical_url = "https://archive-api.open-meteo.com/v1/archive"
# #     end_date = datetime.now()
# #     start_date = end_date - timedelta(days=30)
    
# #     hist_params = {
# #         "latitude": lat,
# #         "longitude": lon,
# #         "start_date": start_date.strftime("%Y-%m-%d"),
# #         "end_date": end_date.strftime("%Y-%m-%d"),
# #         "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
# #         "timezone": "Asia/Karachi"
# #     }
# #     weather_data["historical"] = safe_get(historical_url, hist_params)
    
# #     # NASA POWER (Solar data, humidity)
# #     nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
# #     nasa_params = {
# #         "parameters": "PRECTOTCORR,T2M,RH2M,WS10M",
# #         "community": "ag",
# #         "longitude": lon,
# #         "latitude": lat,
# #         "start": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
# #         "end": datetime.now().strftime("%Y%m%d"),
# #         "format": "json"
# #     }
# #     weather_data["nasa"] = safe_get(nasa_url, nasa_params)
    
# #     return weather_data

# # def fetch_elevation_data(lat, lon, radius_km=50):
# #     elevation_data = {}

# #     # Try USGS
# #     usgs_url = "https://epqs.nationalmap.gov/v1/json"
# #     params = {"x": lon, "y": lat, "wkid": 4326, "units": "Meters"}
# #     elevation_data["usgs"] = safe_get(usgs_url, params)

# #     # Try OpenTopoData
# #     opentopo_url = "https://api.opentopodata.org/v1/srtm90m"
# #     elevation_data["opentopo"] = safe_get(opentopo_url, {"locations": f"{lat},{lon}"})

# #     # Try Open-Elevation
# #     open_elev_url = "https://api.open-elevation.com/api/v1/lookup"
# #     elevation_data["open_elevation"] = safe_get(open_elev_url, {"locations": f"{lat},{lon}"})

# #     return elevation_data


# # # 🌊 River & Water Body Data
# # def fetch_water_bodies(bbox):
# #     """Get rivers, lakes, and water bodies from OpenStreetMap"""
# #     min_lat, min_lon, max_lat, max_lon = bbox
    
# #     # Overpass API query for water features
# #     query = f"""
# #     [out:json][timeout:30];
# #     (
# #       way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
# #     );
# #     out geom tags;
# #     """
    
# #     return safe_post("https://overpass.kumi.systems/api/interpreter", {"data": query})


# # # 🛣️ Road Network & Infrastructure
# # def fetch_infrastructure(bbox):
# #     """Get roads, bridges, and critical infrastructure"""
# #     min_lat, min_lon, max_lat, max_lon = bbox
    
# #     query = f"""
# #     [out:json][timeout:30];
# #     (
# #       way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       way["bridge"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       way["tunnel"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       node["emergency"="assembly_point"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       node["amenity"="hospital"]({min_lat},{min_lon},{max_lat},{max_lon});
# #       node["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
# #     );
# #     out geom tags;
# #     """
    
# #     return safe_post("https://overpass-api.de/api/interpreter", {"data": query})

# # # 🛰️ Satellite & Remote Sensing Data
# # def fetch_satellite_data(lat, lon):
# #     """Get satellite-based flood indicators"""
# #     satellite_data = {}
    
# #     # MODIS Fire/Flood data (NASA)
# #     # Note: This is a simplified example - actual implementation would need proper NASA API
# #     modis_url = "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives"
    
# #     # Sentinel Hub (if available)
# #     # This would require registration but provides excellent flood mapping
    
# #     # Placeholder for now - in production, integrate with:
# #     # - NASA MODIS
# #     # - Sentinel-1 SAR for flood detection
# #     # - Landsat for surface water changes
    
# #     satellite_data["status"] = "APIs require registration - implement based on needs"
# #     return satellite_data

# # # 📊 Flood Risk Indicators
# # def calculate_flood_risk(weather_data, elevation_data, water_bodies):
# #     """Calculate flood risk based on collected data"""
# #     risk_factors = {
# #         "precipitation_risk": 0,
# #         "elevation_risk": 0,
# #         "water_proximity_risk": 0,
# #         "infrastructure_risk": 0,
# #         "overall_risk": "LOW"
# #     }
    
# #     try:
# #         # Precipitation risk
# #         if weather_data and "open_meteo" in weather_data:
# #             forecast = weather_data["open_meteo"]
# #             if forecast and "daily" in forecast:
# #                 precip = forecast["daily"].get("precipitation_sum", [0])
# #                 max_precip = max(precip) if precip else 0
# #                 if max_precip > 50:
# #                     risk_factors["precipitation_risk"] = min(max_precip / 100, 1.0)
        
# #         # Elevation risk (lower elevation = higher risk)
# #         if elevation_data and "srtm" in elevation_data:
# #             srtm = elevation_data["srtm"]
# #             if srtm and "results" in srtm and srtm["results"]:
# #                 elevation = srtm["results"][0].get("elevation", 500)
# #                 if elevation < 100:
# #                     risk_factors["elevation_risk"] = 0.8
# #                 elif elevation < 300:
# #                     risk_factors["elevation_risk"] = 0.4
        
# #         # Water proximity risk
# #         if water_bodies and "elements" in water_bodies:
# #             water_count = len(water_bodies["elements"])
# #             risk_factors["water_proximity_risk"] = min(water_count / 10, 1.0)
        
# #         # Overall risk calculation
# #         avg_risk = np.mean([
# #             risk_factors["precipitation_risk"],
# #             risk_factors["elevation_risk"],
# #             risk_factors["water_proximity_risk"]
# #         ])
        
# #         if avg_risk > 0.7:
# #             risk_factors["overall_risk"] = "HIGH"
# #         elif avg_risk > 0.4:
# #             risk_factors["overall_risk"] = "MEDIUM"
# #         else:
# #             risk_factors["overall_risk"] = "LOW"
            
# #     except Exception as e:
# #         logging.error(f"Risk calculation error: {e}")
    
# #     return risk_factors
# # # 🌱 Soil Moisture & Soil Type
# # def fetch_soil_data(lat, lon):
# #     soil_data = {}

# #     # NASA SMAP soil moisture (via NASA EarthData API → sample)
# #     smap_url = f"https://api.open-meteo.com/v1/forecast"
# #     params = {
# #         "latitude": lat,
# #         "longitude": lon,
# #         "hourly": "soil_moisture_0_7cm",  # Open-Meteo re-exposes SMAP
# #         "timezone": "Asia/Karachi"
# #     }
# #     soil_data["smap"] = safe_get(smap_url, params)

# #     # SoilGrids (ISRIC) - gives soil texture, organic matter etc.
# #     soilgrids_url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
# #     soilgrids_params = {
# #         "lon": lon,
# #         "lat": lat,
# #         "property": ["sand", "silt", "clay", "ocd"],
# #         "depth": "15-30cm"
# #     }
# #     soil_data["soilgrids"] = safe_get(soilgrids_url, soilgrids_params)

# #     return soil_data

# # # 🏞️ Land Use & Runoff
# # def fetch_landuse_data(lat, lon, radius_km=5):
# #     landuse_data = {}

# #     # ESA WorldCover (10m land cover) - free static dataset
# #     esa_url = f"https://services.terrascope.be/wms"
# #     # (requires WMS request, for now mark placeholder)
# #     landuse_data["esa_worldcover"] = "WMS download required (ESA WorldCover)"

# #     # Sentinel-2 NDVI/NDWI (for vegetation & water extent)
# #     landuse_data["sentinel2"] = "Integrate via Google Earth Engine (needs token)"

# #     return landuse_data
# # # 📜 Historical Flood Data
# # def fetch_historical_floods(lat, lon):
# #     history_data = {}

# #     # UNOSAT Flood Portal (shapefiles, GeoJSON) - needs parsing
# #     history_data["unosat"] = "Requires UNOSAT GeoJSON download"

# #     # GFMS Past Flood Events
# #     gfms_hist_url = f"https://floods.gsfc.nasa.gov/api/v1.0/events"
# #     params = {"lat": lat, "lon": lon, "years": 10}
# #     history_data["gfms_events"] = safe_get(gfms_hist_url, params)

# #     return history_data


# # # 🤖 AI Analysis with Gemini
# # def gemini_analyze_flood_risk(all_data):
# #     """Use Gemini to provide intelligent flood risk analysis"""
# #     prompt = f"""
# # Flood Risk Analysis for Pakistan - Data Analysis:

# # WEATHER DATA: {json.dumps(all_data.get('weather', {}), indent=1)[:2000]}
# # ELEVATION: {json.dumps(all_data.get('elevation', {}), indent=1)[:1000]}
# # WATER BODIES: {json.dumps(all_data.get('water_bodies', {}), indent=1)[:1000]}
# # RISK CALCULATION: {json.dumps(all_data.get('risk_assessment', {}), indent=1)}

# # Please provide a comprehensive flood risk analysis including:

# # 1. **Risk Level**: HIGH/MEDIUM/LOW with reasoning
# # 2. **Key Factors**: Most important contributing factors
# # 3. **Timeline**: When is risk highest (next 24h, 48h, week)
# # 4. **Recommendations**: Specific actionable advice
# # 5. **Urdu Summary**: 2-3 lines in Urdu for local communication
# # 6. **Critical Areas**: Specific locations most at risk

# # Format as structured JSON with these exact keys:
# # - risk_level
# # - key_factors (array)
# # - timeline
# # - recommendations (array)
# # - urdu_summary
# # - critical_areas (array)
# # """
    
# #     try:
# #         response = client.models.generate_content(
# #             model="gemini-2.0-flash",
# #             contents=prompt
# #         )
        
# #         # Try to extract JSON from response
# #         response_text = response.text
# #         if "```json" in response_text:
# #             json_start = response_text.find("```json") + 7
# #             json_end = response_text.find("```", json_start)
# #             json_str = response_text[json_start:json_end].strip()
# #             return json.loads(json_str)
# #         else:
# #             return {"analysis": response_text, "format": "text"}
            
# #     except Exception as e:
# #         logging.error(f"Gemini analysis failed: {e}")
# #         return {"error": str(e), "raw_data_available": True}

# # # -----------------------------
# # # 4️⃣ Main Data Collection Orchestrator
# # # -----------------------------
# # def collect_flood_data(bbox, target_area_name="Unknown"):
# #     """Main function to collect all flood-related data"""
    
# #     # Calculate center point
# #     lat = (bbox[0] + bbox[2]) / 2
# #     lon = (bbox[1] + bbox[3]) / 2
    
# #     logging.info(f"🌊 Starting flood data collection for {target_area_name}")
    
# #     # Initialize output structure
# #     output ={
# #   "meta": {
# #     "area_name": "Karachi Metropolitan",
# #     "bbox": [24.8, 66.9, 25.3, 67.4],
# #     "center": [25.05, 67.15],
# #     "timestamp": 1758723041,
# #     "datetime": "2025-09-24T19:40:07",
# #     "agent": "PakistanFloodDataAgent",
# #     "version": "2.0",
# #     "role": "Comprehensive flood forecasting data collection and risk assessment"
# #   },
# #   "data_sources": {
# #     "soil": {...},
# #     "landuse": {...},
# #     "river": {...},
# #     "historical": {...},
# #     "satellite_flood": {...}
# #   },
# #   "risk_assessment": {},
# #   "ai_analysis": {},
# #   "status": "processing"
# # }

    
# #     # 1. Weather Data Collection
# #     logging.info("📡 Fetching weather data...")
# #     output["data_sources"]["weather"] = fetch_weather_data(lat, lon)
    
# #     # 2. Elevation Analysis  
# #     logging.info("🏔️ Fetching elevation data...")
# #     output["data_sources"]["elevation"] = fetch_elevation_data(lat, lon)
    
# #     # 3. Water Bodies
# #     logging.info("🌊 Fetching water bodies...")
# #     output["data_sources"]["water_bodies"] = fetch_water_bodies(bbox)
    
# #     # 4. Infrastructure
# #     logging.info("🛣️ Fetching infrastructure data...")
# #     output["data_sources"]["infrastructure"] = fetch_infrastructure(bbox)
    
# #     # 5. Satellite Data (placeholder)
# #     logging.info("🛰️ Fetching satellite data...")
# #     output["data_sources"]["satellite"] = fetch_satellite_data(lat, lon)
    
# #     # 6. Risk Assessment
# #     logging.info("📊 Calculating flood risk...")
# #     output["risk_assessment"] = calculate_flood_risk(
# #         output["data_sources"]["weather"],
# #         output["data_sources"]["elevation"],
# #         output["data_sources"]["water_bodies"]
# #     )
    
# #     # 7. AI Analysis
# #     logging.info("🤖 Running AI analysis...")
# #     output["ai_analysis"] = gemini_analyze_flood_risk(output["data_sources"])
    
# #     # 8. Final status
# #     output["status"] = "complete"
# #     output["meta"]["completion_time"] = int(time.time())
    
# #     logging.info("✅ Flood data collection complete!")
# #     return output
# #     # 6. Soil & Land Data
# #     logging.info("🌱 Fetching soil and land data...")
# #     output["data_sources"]["soil"] = fetch_soil_data(lat, lon)
# #     output["data_sources"]["landuse"] = fetch_landuse_data(lat, lon)

# #     # 7. River Flow Data
# #     logging.info("🌊 Fetching river flow/discharge data...")
# #     output["data_sources"]["river"] = fetch_river_data(lat, lon)

# #     # 8. Historical Flood Data
# #     logging.info("📜 Fetching historical flood events...")
# #     output["data_sources"]["historical"] = fetch_historical_floods(lat, lon)

# #     # 9. Satellite Flood Extent
# #     logging.info("🛰️ Fetching satellite flood extent...")
# #     output["data_sources"]["satellite_flood"] = fetch_satellite_flood(lat, lon)


# # # -----------------------------
# # # 5️⃣ Pakistan-specific presets
# # # -----------------------------
# # PAKISTAN_AREAS = {
# #     "karachi": {
# #         "bbox": [24.7, 66.9, 25.2, 67.5],
# #         "name": "Karachi Metropolitan"
# #     },
# #     "lahore": {
# #         "bbox": [31.4, 74.2, 31.7, 74.5],
# #         "name": "Lahore District"
# #     },
# #     "islamabad": {
# #         "bbox": [33.6, 72.8, 33.8, 73.2],
# #         "name": "Islamabad Capital Territory"
# #     },
# #     "faisalabad": {
# #         "bbox": [31.3, 73.0, 31.5, 73.2],
# #         "name": "Faisalabad District"
# #     },
# #     "rawalpindi": {
# #         "bbox": [33.5, 73.0, 33.7, 73.2],
# #         "name": "Rawalpindi District"
# #     },
# #     "multan": {
# #         "bbox": [30.1, 71.4, 30.3, 71.6],
# #         "name": "Multan District"
# #     },
# #     "peshawar": {
# #         "bbox": [34.0, 71.5, 34.2, 71.7],
# #         "name": "Peshawar District"
# #     },
# #     "quetta": {
# #         "bbox": [30.1, 66.9, 30.3, 67.1],
# #         "name": "Quetta District"
# #     }
# # }

# # # -----------------------------
# # # 6️⃣ Main Execution
# # # -----------------------------
# # if __name__ == "__main__":
 
# # # -----------------------------
# #     print("\n🌍 Select target area for flood risk analysis:\n")
# #     for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
# #         print(f"{i}. {val['name']} ({key})")
# #     print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")

# #     choice = int(input("\nEnter choice number: "))

# #     if 1 <= choice <= len(PAKISTAN_AREAS):
# #         area_key = list(PAKISTAN_AREAS.keys())[choice-1]
# #         area_config = PAKISTAN_AREAS[area_key]
# #         bbox = area_config["bbox"]
# #         name = area_config["name"]
# #     else:
# #         # Custom Area
# #         print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
# #         bbox = list(map(float, input("Format: lat1,lon1,lat2,lon2 → ").split(",")))
# #         name = "Custom Area"

# #     # Collect flood data
# #     flood_data = collect_flood_data(bbox, name)


# # # # data_agent_final.py
# # # import os
# # # import time
# # # import requests
# # # import logging
# # # from dotenv import load_dotenv
# # # from google import genai

# # # # -----------------------------
# # # # 1️⃣ Load environment & keys
# # # # -----------------------------
# # # load_dotenv()
# # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # if not GEMINI_API_KEY:
# # #     raise RuntimeError("❌ GEMINI_API_KEY not found. Check your .env file!")

# # # client = genai.Client(api_key=GEMINI_API_KEY)

# # # logging.basicConfig(level=logging.INFO)

# # # # -----------------------------
# # # # 2️⃣ Helper Functions
# # # # -----------------------------
# # # def safe_get(url, params=None, timeout=15):
# # #     try:
# # #         r = requests.get(url, params=params, timeout=timeout)
# # #         r.raise_for_status()
# # #         return r.json()
# # #     except Exception as e:
# # #         logging.warning(f"GET failed ({url}): {e}")
# # #         return None

# # # # -----------------------------
# # # # 3️⃣ Data Sources
# # # # -----------------------------

# # # # Weather & Rainfall (Open-Meteo)
# # # def fetch_weather(lat, lon):
# # #     url = "https://api.open-meteo.com/v1/forecast"
# # #     params = {
# # #         "latitude": lat,
# # #         "longitude": lon,
# # #         "hourly": "temperature_2m,precipitation,relative_humidity_2m"
# # #     }
# # #     return safe_get(url, params)

# # # # Rivers / Flood Levels (PMD or open source)
# # # def fetch_river_levels():
# # #     # placeholder open-access API
# # #     url = "https://api.data.gov.pk/v1/flood-levels"  # fake example
# # #     return safe_get(url)

# # # # NDMA / Relief camps
# # # def fetch_ndma_camps():
# # #     # placeholder open-access API
# # #     url = "https://api.data.gov.pk/v1/ndma-camps"  # fake example
# # #     return safe_get(url)

# # # # OSM Road closures / flooded roads
# # # def fetch_osm_closures(bbox):
# # #     min_lat, min_lon, max_lat, max_lon = bbox
# # #     query = f"""
# # #     [out:json][timeout:25];
# # #     (
# # #       way["highway"]({min_lat},{min_lon},{max_lat},{max_lon})["construction"];
# # #       way["highway"]({min_lat},{min_lon},{max_lat},{max_lon})["surface"~"flooded|mud|water"];
# # #     );
# # #     out geom tags;
# # #     """
# # #     try:
# # #         r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
# # #         r.raise_for_status()
# # #         return r.json()
# # #     except Exception as e:
# # #         logging.warning(f"Overpass failed: {e}")
# # #         return None

# # # # DEM / Terrain (Optional)
# # # def get_elevation(lat, lon, dem_path="srtm.tif"):
# # #     try:
# # #         import rasterio
# # #         with rasterio.open(dem_path) as dem:
# # #             row, col = dem.index(lon, lat)
# # #             elevation = dem.read(1)[row, col]
# # #             return elevation
# # #     except Exception as e:
# # #         logging.warning(f"DEM failed: {e}")
# # #         return None

# # # # Gemini summarizer
# # # def gemini_normalize(raw):
# # #     prompt = f"""
# # # Flood + weather + rivers + OSM + NDMA data:
# # # {raw}
# # # Extract structured JSON:
# # # - events: list of {{type, severity, lat, lon, description}}
# # # - affected_places
# # # - relief_camps
# # # - summary_urdu: 2 lines only
# # # """
# # #     response = client.models.generate_content(
# # #         model="gemini-2.0-flash",
# # #         input=prompt
# # #     )
# # #     return response.output_text

# # # # -----------------------------
# # # # 4️⃣ Orchestrator
# # # # -----------------------------
# # # def collect_data(bbox):
# # #     lat = (bbox[0] + bbox[2]) / 2
# # #     lon = (bbox[1] + bbox[3]) / 2
# # #     out = {
# # #         "meta": {
# # #             "bbox": bbox,
# # #             "center": [lat, lon],
# # #             "time": int(time.time()),
# # #             "agent": "DataAgent",       # ✅ Role added
# # #             "role": "Collects and categorizes flood, weather, terrain, and road closure data"
# # #         },
# # #         "categories": {}
# # #     }

# # #     # Data fetching
# # #     out["categories"]["weather"] = fetch_weather(lat, lon)
# # #     out["categories"]["osm"] = fetch_osm_closures(bbox)
# # #     # Add more categories as needed

# # #     return out

# # # # -----------------------------
# # # # 5️⃣ Main
# # # # -----------------------------
# # # if __name__ == "__main__":
# # #     bbox = (24.8, 67.0, 25.2, 67.5)  # Karachi example
# # #     data = collect_data(bbox)

# # #     import json
# # #     print(json.dumps(data, indent=2, ensure_ascii=False)[:5000])  # first 5k chars





# import os
# import time
# import requests
# import logging
# import json
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# from google import genai
# import numpy as np

# # -----------------------------
# # 1️⃣ Load environment & keys
# # -----------------------------
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("❌ GEMINI_API_KEY not found. Check your .env file!")

# client = genai.Client(api_key=GEMINI_API_KEY)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # -----------------------------
# # 2️⃣ Cache for API Responses
# # -----------------------------
# api_cache = {}  # Simple dictionary-based cache
# CACHE_TTL = 3600  # 1-hour TTL

# def get_cached_response(cache_key):
#     if cache_key in api_cache:
#         timestamp, data = api_cache[cache_key]
#         if time.time() - timestamp < CACHE_TTL:
#             return data
#     return None

# def set_cached_response(cache_key, data):
#     api_cache[cache_key] = (time.time(), data)

# # -----------------------------
# # 3️⃣ Helper Functions
# # -----------------------------
# def safe_get(url, params=None, timeout=15, headers=None, retries=3):
#     """Safe HTTP GET with error handling and caching"""
#     cache_key = (url, str(params))
#     cached = get_cached_response(cache_key)
#     if cached:
#         return cached

#     headers = headers or {'User-Agent': 'FloodForecast/1.0'}
#     for attempt in range(retries):
#         try:
#             r = requests.get(url, params=params, timeout=timeout, headers=headers)
#             r.raise_for_status()
#             result = r.json()
#             set_cached_response(cache_key, result)
#             return result
#         except Exception as e:
#             logging.warning(f"GET failed ({url}, attempt {attempt+1}/{retries}): {e}")
#             if attempt < retries - 1:
#                 time.sleep(2 ** attempt)  # Exponential backoff
#     return None

# def safe_post(url, data=None, timeout=60, headers=None, retries=3):  # Increased timeout for Overpass
#     """Safe HTTP POST with error handling and caching"""
#     cache_key = (url, str(data))
#     cached = get_cached_response(cache_key)
#     if cached:
#         return cached

#     headers = headers or {'User-Agent': 'FloodForecast/1.0'}
#     for attempt in range(retries):
#         try:
#             r = requests.post(url, data=data, timeout=timeout, headers=headers)
#             r.raise_for_status()
#             result = r.json()
#             set_cached_response(cache_key, result)
#             return result
#         except Exception as e:
#             logging.warning(f"POST failed ({url}, attempt {attempt+1}/{retries}): {e}")
#             if attempt < retries - 1:
#                 time.sleep(2 ** attempt)
#     return None

# # -----------------------------
# # 4️⃣ Enhanced Data Sources
# # -----------------------------

# # 🌦️ Weather & Rainfall (Multiple sources)
# def fetch_weather_data(lat, lon):
#     """Comprehensive weather data from multiple sources"""
#     weather_data = {}
    
#     # Open-Meteo (Primary - very reliable)
#     open_meteo_url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", 
#                   "wind_speed_10m", "pressure_msl", "cloud_cover"],
#         "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
#                  "wind_speed_10m_max", "precipitation_probability_max"],
#         "timezone": "Asia/Karachi",
#         "forecast_days": 7
#     }
#     weather_data["open_meteo"] = safe_get(open_meteo_url, params)
    
#     # Historical weather for context
#     historical_url = "https://archive-api.open-meteo.com/v1/archive"
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=30)
    
#     hist_params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": start_date.strftime("%Y-%m-%d"),
#         "end_date": end_date.strftime("%Y-%m-%d"),
#         "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
#         "timezone": "Asia/Karachi"
#     }
#     weather_data["historical"] = safe_get(historical_url, hist_params)
    
#     # NASA POWER (Solar data, humidity)
#     nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
#     nasa_params = {
#         "parameters": "PRECTOTCORR,T2M,RH2M,WS10M",
#         "community": "ag",
#         "longitude": lon,
#         "latitude": lat,
#         "start": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
#         "end": datetime.now().strftime("%Y%m%d"),
#         "format": "json"
#     }
#     weather_data["nasa"] = safe_get(nasa_url, nasa_params)
    
#     return weather_data

# def fetch_elevation_data(lat, lon, radius_km=50):
#     """Fetch elevation data with retries - Skip USGS for non-US locations"""
#     elevation_data = {}

#     # Skip USGS for Pakistan (non-US)
#     elevation_data["usgs"] = None  # Would fail for international locations

#     # Try OpenTopoData
#     opentopo_url = "https://api.opentopodata.org/v1/srtm90m"
#     elevation_data["opentopo"] = safe_get(opentopo_url, {"locations": f"{lat},{lon}"})

#     # Try Open-Elevation
#     open_elev_url = "https://api.open-elevation.com/api/v1/lookup"
#     elevation_data["open_elevation"] = safe_get(open_elev_url, {"locations": f"{lat},{lon}"})

#     return elevation_data

# # 🌊 River & Water Body Data
# def fetch_water_bodies(bbox):
#     """Get rivers, lakes, and water bodies from OpenStreetMap - Use reliable server"""
#     min_lat, min_lon, max_lat, max_lon = bbox
    
#     query = f"""
#     [out:json][timeout:180];
#     (
#       way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
#       way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
#       relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
#       way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
#     );
#     out geom tags;
#     """
    
#     # Switch to overpass-api.de for reliability
#     return safe_post("https://overpass-api.de/api/interpreter", {"data": query})

# # 🛣️ Road Network & Infrastructure
# def fetch_infrastructure(bbox):
#     """Get major roads and critical infrastructure"""
#     min_lat, min_lon, max_lat, max_lon = bbox
    
#     query = f"""
#     [out:json][timeout:30];
#     (
#       way["highway"~"motorway|trunk|primary"]({min_lat},{min_lon},{max_lat},{max_lon});
#       node["amenity"="hospital"]({min_lat},{min_lon},{max_lat},{max_lon});
#     );
#     out geom tags;
#     """
    
#     return safe_post("https://overpass-api.de/api/interpreter", {"data": query})

# # 🛰️ Satellite & Remote Sensing Data
# def fetch_satellite_data(lat, lon):
#     """Get satellite-based flood indicators"""
#     satellite_data = {}
    
#     # MODIS Fire/Flood data (NASA)
#     modis_url = "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives"
#     satellite_data["modis"] = "APIs require registration - implement based on needs"
    
#     return satellite_data

# # 🌱 Soil Moisture & Soil Type
# def fetch_soil_data(lat, lon):
#     """Fetch soil moisture and type data - Use archive API for soil moisture"""
#     soil_data = {}

#     # NASA SMAP soil moisture via Open-Meteo Archive
#     smap_url = "https://archive-api.open-meteo.com/v1/archive"
#     start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
#     end_date = datetime.now().strftime("%Y-%m-%d")
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": start_date,
#         "end_date": end_date,
#         "hourly": "soil_moisture_0_to_7cm",  # Correct parameter name
#         "timezone": "Asia/Karachi"
#     }
#     soil_data["smap"] = safe_get(smap_url, params)

#     # SoilGrids (ISRIC)
#     soilgrids_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
#     soilgrids_params = {
#         "lon": lon,
#         "lat": lat,
#         "property": ["sand", "silt", "clay", "ocd"],
#         "depth": "15-30cm"
#     }
#     soil_data["soilgrids"] = safe_get(soilgrids_url, soilgrids_params)

#     return soil_data

# # 🏞️ Land Use & Runoff
# def fetch_landuse_data(lat, lon, radius_km=5):
#     """Fetch land use and runoff data"""
#     landuse_data = {}

#     # ESA WorldCover
#     landuse_data["esa_worldcover"] = "WMS download required (ESA WorldCover)"

#     # Sentinel-2 NDVI/NDWI
#     landuse_data["sentinel2"] = "Integrate via Google Earth Engine (needs token)"

#     return landuse_data

# # 📜 Historical Flood Data
# def fetch_historical_floods(lat, lon):
#     """Fetch historical flood data - GFMS API not publicly available, use placeholder"""
#     history_data = {}

#     # UNOSAT Flood Portal
#     history_data["unosat"] = "Requires UNOSAT GeoJSON download"

#     # GFMS - No public REST API, placeholder
#     history_data["gfms_events"] = {"status": "No public API available; check http://flood.umd.edu/ for datasets"}

#     return history_data

# # 📊 Flood Risk Indicators
# def calculate_flood_risk(weather_data, elevation_data, water_bodies):
#     """Calculate flood risk based on collected data"""
#     risk_factors = {
#         "precipitation_risk": 0,
#         "elevation_risk": 0,
#         "water_proximity_risk": 0,
#         "infrastructure_risk": 0,
#         "overall_risk": "LOW"
#     }
    
#     try:
#         # Precipitation risk
#         if weather_data and "open_meteo" in weather_data:
#             forecast = weather_data["open_meteo"]
#             if forecast and "daily" in forecast:
#                 precip = forecast["daily"].get("precipitation_sum", [0])
#                 max_precip = max(precip) if precip else 0
#                 if max_precip > 50:
#                     risk_factors["precipitation_risk"] = min(max_precip / 100, 1.0)
        
#         # Elevation risk (prioritize open_elevation)
#         elevation_sources = ["open_elevation", "opentopo"]
#         elevation = 500  # default
#         for source in elevation_sources:
#             if elevation_data and source in elevation_data:
#                 data = elevation_data[source]
#                 if data and "results" in data and data["results"]:
#                     elevation = data["results"][0].get("elevation", 500)
#                     break
#                 if data and isinstance(data, dict) and "elevation" in data:
#                     elevation = data["elevation"]
#                     break
        
#                 if elevation < 100:
#                     risk_factors["elevation_risk"] = 0.8
#                 elif elevation < 300:
#                     risk_factors["elevation_risk"] = 0.4
        
#         # Water proximity risk
#         if water_bodies and "elements" in water_bodies:
#             water_count = len(water_bodies["elements"])
#             risk_factors["water_proximity_risk"] = min(water_count / 10, 1.0)
        
#         # Overall risk calculation
#         avg_risk = np.mean([
#             risk_factors["precipitation_risk"],
#             risk_factors["elevation_risk"],
#             risk_factors["water_proximity_risk"]
#         ])
        
#         if avg_risk > 0.7:
#             risk_factors["overall_risk"] = "HIGH"
#         elif avg_risk > 0.4:
#             risk_factors["overall_risk"] = "MEDIUM"
#         else:
#             risk_factors["overall_risk"] = "LOW"
            
#     except Exception as e:
#         logging.error(f"Risk calculation error: {e}")
    
#     return risk_factors

# # 🤖 AI Analysis with Gemini
# def gemini_analyze_flood_risk(all_data):
#     """Use Gemini to provide intelligent flood risk analysis"""
#     prompt = f"""
# Flood Risk Analysis for Pakistan - Data Analysis:

# WEATHER DATA: {json.dumps(all_data.get('weather', {}), indent=1)[:2000]}
# ELEVATION: {json.dumps(all_data.get('elevation', {}), indent=1)[:1000]}
# WATER BODIES: {json.dumps(all_data.get('water_bodies', {}), indent=1)[:1000]}
# RISK CALCULATION: {json.dumps(all_data.get('risk_assessment', {}), indent=1)}

# Please provide a comprehensive flood risk analysis including:

# 1. **Risk Level**: HIGH/MEDIUM/LOW with reasoning
# 2. **Key Factors**: Most important contributing factors
# 3. **Timeline**: When is risk highest (next 24h, 48h, week)
# 4. **Recommendations**: Specific actionable advice
# 5. **Urdu Summary**: 2-3 lines in Urdu for local communication
# 6. **Critical Areas**: Specific locations most at risk

# Format as structured JSON with these exact keys:
# - risk_level
# - key_factors (array)
# - timeline
# - recommendations (array)
# - urdu_summary
# - critical_areas (array)
# """
    
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=prompt
#         )
        
#         # Try to extract JSON from response
#         response_text = response.text
#         if "```json" in response_text:
#             json_start = response_text.find("```json") + 7
#             json_end = response_text.find("```", json_start)
#             json_str = response_text[json_start:json_end].strip()
#             try:
#                 return json.loads(json_str)
#             except json.JSONDecodeError as e:
#                 logging.error(f"Failed to parse Gemini JSON: {e}")
#                 return {"error": "Invalid JSON response", "raw_data": response_text}
#         else:
#             logging.warning("Gemini response not in JSON format")
#             return {"analysis": response_text, "format": "text"}
            
#     except Exception as e:
#         logging.error(f"Gemini analysis failed: {e}")
#         return {"error": str(e), "raw_data_available": True}

# # -----------------------------
# # 5️⃣ Main Data Collection Orchestrator
# # -----------------------------
# def collect_flood_data(bbox, target_area_name="Unknown"):
#     """Main function to collect all flood-related data"""
    
#     # Calculate center point
#     lat = (bbox[0] + bbox[2]) / 2
#     lon = (bbox[1] + bbox[3]) / 2
    
#     logging.info(f"🌊 Starting flood data collection for {target_area_name}")
    
#     # Initialize output structure
#     output = {
#         "meta": {
#             "area_name": target_area_name,
#             "bbox": bbox,
#             "center": [lat, lon],
#             "timestamp": int(time.time()),
#             "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
#             "agent": "PakistanFloodDataAgent",
#             "version": "2.2",
#             "role": "Comprehensive flood forecasting data collection and risk assessment"
#         },
#         "data_sources": {
#             "weather": {},
#             "elevation": {},
#             "water_bodies": {},
#             "infrastructure": {},
#             "satellite": {},
#             "soil": {},
#             "landuse": {},
#             "river": {},
#             "historical": {},
#             "satellite_flood": {}
#         },
#         "risk_assessment": {},
#         "ai_analysis": {},
#         "visualization": {},
#         "status": "processing"
#     }
    
#     # 1. Weather Data Collection
#     logging.info("📡 Fetching weather data...")
#     output["data_sources"]["weather"] = fetch_weather_data(lat, lon)
    
#     # 2. Elevation Analysis  
#     logging.info("🏔️ Fetching elevation data...")
#     output["data_sources"]["elevation"] = fetch_elevation_data(lat, lon)
    
#     # 3. Water Bodies
#     logging.info("🌊 Fetching water bodies...")
#     output["data_sources"]["water_bodies"] = fetch_water_bodies(bbox)
    
#     # 4. Infrastructure
#     logging.info("🛣️ Fetching infrastructure data...")
#     output["data_sources"]["infrastructure"] = fetch_infrastructure(bbox)
    
#     # 5. Satellite Data
#     logging.info("🛰️ Fetching satellite data...")
#     output["data_sources"]["satellite"] = fetch_satellite_data(lat, lon)
    
#     # 6. Soil & Land Data
#     logging.info("🌱 Fetching soil and land data...")
#     output["data_sources"]["soil"] = fetch_soil_data(lat, lon)
#     output["data_sources"]["landuse"] = fetch_landuse_data(lat, lon)
    
#     # 7. River Flow Data (Placeholder)
#     logging.info("🌊 Fetching river flow/discharge data...")
#     output["data_sources"]["river"] = {"status": "Placeholder - implement river flow API"}
    
#     # 8. Historical Flood Data
#     logging.info("📜 Fetching historical flood events...")
#     output["data_sources"]["historical"] = fetch_historical_floods(lat, lon)
    
#     # 9. Satellite Flood Extent (Placeholder)
#     logging.info("🛰️ Fetching satellite flood extent...")
#     output["data_sources"]["satellite_flood"] = {"status": "Placeholder - implement satellite flood API"}
    
#     # 10. Risk Assessment
#     logging.info("📊 Calculating flood risk...")
#     output["risk_assessment"] = calculate_flood_risk(
#         output["data_sources"]["weather"],
#         output["data_sources"]["elevation"],
#         output["data_sources"]["water_bodies"]
#     )
    
#     # 11. AI Analysis
#     logging.info("🤖 Running AI analysis...")
#     output["ai_analysis"] = gemini_analyze_flood_risk(output["data_sources"])
    
#     # 12. Visualization (Precipitation Forecast)
#     if "open_meteo" in output["data_sources"]["weather"]:
#         precip = output["data_sources"]["weather"]["open_meteo"].get("daily", {}).get("precipitation_sum", [0] * 7)
#         days = list(range(1, len(precip) + 1))
#         output["visualization"] = {
#             "type": "line",
#             "data": {
#                 "labels": days,
#                 "datasets": [{
#                     "label": "Daily Precipitation (mm)",
#                     "data": precip,
#                     "borderColor": "#1e90ff",
#                     "backgroundColor": "rgba(30, 144, 255, 0.2)",
#                     "fill": True
#                 }]
#             },
#             "options": {
#                 "scales": {
#                     "y": {"title": {"display": True, "text": "Precipitation (mm)"}},
#                     "x": {"title": {"display": True, "text": "Day"}}
#                 }
#             }
#         }
    
#     # 13. Final status
#     output["status"] = "complete"
#     output["meta"]["completion_time"] = int(time.time())
    
#     logging.info("✅ Flood data collection complete!")
#     return output

# # -----------------------------
# # 6️⃣ Pakistan-specific presets
# # -----------------------------
# PAKISTAN_AREAS = {
#     "karachi": {
#         "bbox": [24.7, 66.9, 25.2, 67.5],
#         "name": "Karachi Metropolitan"
#     },
#     "lahore": {
#         "bbox": [31.4, 74.2, 31.7, 74.5],
#         "name": "Lahore District"
#     },
#     "islamabad": {
#         "bbox": [33.6, 72.8, 33.8, 73.2],
#         "name": "Islamabad Capital Territory"
#     },
#     "faisalabad": {
#         "bbox": [31.3, 73.0, 31.5, 73.2],
#         "name": "Faisalabad District"
#     },
#     "rawalpindi": {
#         "bbox": [33.5, 73.0, 33.7, 73.2],
#         "name": "Rawalpindi District"
#     },
#     "multan": {
#         "bbox": [30.1, 71.4, 30.3, 71.6],
#         "name": "Multan District"
#     },
#     "peshawar": {
#         "bbox": [34.0, 71.5, 34.2, 71.7],
#         "name": "Peshawar District"
#     },
#     "quetta": {
#         "bbox": [30.1, 66.9, 30.3, 67.1],
#         "name": "Quetta District"
#     }
# }

# # -----------------------------
# # 7️⃣ Main Execution
# # -----------------------------
# if __name__ == "__main__":
#     print("\n🌍 Select target area for flood risk analysis:\n")
#     for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
#         print(f"{i}. {val['name']} ({key})")
#     print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")

#     choice = int(input("\nEnter choice number: "))

#     if 1 <= choice <= len(PAKISTAN_AREAS):
#         area_key = list(PAKISTAN_AREAS.keys())[choice-1]
#         area_config = PAKISTAN_AREAS[area_key]
#         bbox = area_config["bbox"]
#         name = area_config["name"]
#     else:
#         print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
#         bbox = list(map(float, input("Format: lat1,lon1,lat2,lon2 → ").split(",")))
#         name = "Custom Area"

#     # Collect flood data
#     flood_data = collect_flood_data(bbox, name)
    
#     # Print AI analysis results
#     print("\n📊 AI Analysis Results:")
#     print(json.dumps(flood_data["ai_analysis"], indent=2, ensure_ascii=False))
    
#     # Print visualization configuration
#     print("\n📈 Visualization Configuration (Chart.js):")
#     print(json.dumps(flood_data["visualization"], indent=2))




import os
import time
import requests
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
import numpy as np
from datetime import datetime, timedelta
# -----------------------------
# 1️⃣ Load environment & keys
# -----------------------------
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("❌ GEMINI_API_KEY not found. Check your .env file!")

# client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# 2️⃣ Cache for API Responses
# -----------------------------
api_cache = {}  # Simple dictionary-based cache
CACHE_TTL = 3600  # 1-hour TTL

def get_cached_response(cache_key):
    if cache_key in api_cache:
        timestamp, data = api_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_cached_response(cache_key, data):
    api_cache[cache_key] = (time.time(), data)

# -----------------------------
# 3️⃣ Helper Functions
# -----------------------------
def safe_get(url, params=None, timeout=15, headers=None, retries=3):
    """Safe HTTP GET with error handling and caching"""
    cache_key = (url, str(params))
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
                time.sleep(2 ** attempt)  # Exponential backoff
    return None

def safe_post(url, data=None, timeout=60, headers=None, retries=3):
    """Safe HTTP POST with error handling and caching"""
    cache_key = (url, str(data))
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
    return None

# -----------------------------
# 4️⃣ Enhanced Data Sources
# -----------------------------

# 🌦️ Weather & Rainfall (Multiple sources)
def fetch_weather_data(lat, lon):
    """Comprehensive weather data from multiple sources"""
    weather_data = {}
    
    # Open-Meteo (Primary - very reliable)
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", 
                  "wind_speed_10m", "pressure_msl", "cloud_cover"],
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
                 "wind_speed_10m_max", "precipitation_probability_max"],
        "timezone": "Asia/Karachi",
        "forecast_days": 7
    }
    weather_data["open_meteo"] = safe_get(open_meteo_url, params)
    
    # Historical weather for context
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
    weather_data["historical"] = safe_get(historical_url, hist_params)
    
    # NASA POWER (Solar data, humidity)
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
    weather_data["nasa"] = safe_get(nasa_url, nasa_params)
    
    return weather_data

def fetch_elevation_data(lat, lon, radius_km=50):
    """Fetch elevation data with retries - Skip USGS for non-US locations"""
    elevation_data = {}

    # Skip USGS for Pakistan (non-US)
    elevation_data["usgs"] = None  # Would fail for international locations

    # Try OpenTopoData
    opentopo_url = "https://api.opentopodata.org/v1/srtm90m"
    elevation_data["opentopo"] = safe_get(opentopo_url, {"locations": f"{lat},{lon}"})

    # Try Open-Elevation
    open_elev_url = "https://api.open-elevation.com/api/v1/lookup"
    elevation_data["open_elevation"] = safe_get(open_elev_url, {"locations": f"{lat},{lon}"})

    return elevation_data

# 🌊 River & Water Body Data
def fetch_water_bodies(bbox):
    """Get rivers, lakes, and water bodies from OpenStreetMap - Use reliable server"""
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
    
    return safe_post("https://overpass-api.de/api/interpreter", {"data": query})

# 🛣️ Road Network & Infrastructure
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
    
    return safe_post("https://overpass-api.de/api/interpreter", {"data": query})

# 🛰️ Satellite & Remote Sensing Data
def fetch_satellite_data(lat, lon):
    """Get satellite-based flood indicators"""
    satellite_data = {}
    
    # Placeholder for Sentinel Hub (free tier available)
    satellite_data["sentinel_hub"] = "Register at https://www.sentinel-hub.com/ for Sentinel-2 flood extent data"
    
    return satellite_data

# Fix SoilGrids in fetch_soil_data
def fetch_soil_data(lat, lon):
    """Fetch soil moisture from Open-Meteo and use static soil type"""
    soil_data = {}
    # Fetch soil moisture from Open-Meteo
    logging.info("🌱 Fetching soil moisture data from Open-Meteo...")
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
    soil_data["soil_moisture"] = safe_get(smap_url, params)
    # Static soil type for urban areas
    logging.info("🌱 Using default urban soil profile for soil type...")
    soil_data["soil_type"] = {
        "sand": 40,  # Percentage
        "silt": 30,
        "clay": 25,
        "organic_carbon_density": 0.5,  # g/kg
        "note": "Default urban soil profile (sand-heavy, moderate drainage)"
    }
    return soil_data
# 🏞️ Land Use & Runoff
def fetch_landuse_data(lat, lon, radius_km=5):
    """Fetch land use and runoff data"""
    landuse_data = {}

    # ESA WorldCover
    landuse_data["esa_worldcover"] = "WMS download required (ESA WorldCover)"

    # Sentinel-2 NDVI/NDWI
    landuse_data["sentinel2"] = "Integrate via Google Earth Engine or Sentinel Hub (needs token)"

    return landuse_data

# 📜 Historical Flood Data
def fetch_historical_floods(lat, lon):
    """Fetch historical flood data"""
    history_data = {}

    # UNOSAT Flood Portal
    history_data["unosat"] = "Download GeoJSON from https://unosat.org/products/flood"

    # Dartmouth Flood Observatory
    history_data["dartmouth"] = "Check http://floodobservatory.colorado.edu/ for historical flood records"

    return history_data
def fetch_river_data(lat, lon):
    """Fetch river flow/discharge data (placeholder for GloFAS)"""
    logging.info("🌊 Fetching river flow/discharge data...")
    river_data = {
        "glofas": {
            "status": "Placeholder - download NetCDF from https://data.jrc.ec.europa.eu/collection/glofas",
            "note": "GloFAS provides river discharge forecasts; implement parsing with xarray for specific rivers"
        }
    }
    return river_data

# Fix Gemini JSON parsing in gemini_analyze_flood_risk

# 📊 Flood Risk Indicators
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
        # Precipitation risk
        if weather_data and "open_meteo" in weather_data:
            forecast = weather_data["open_meteo"]
            if forecast and "daily" in forecast:
                precip = forecast["daily"].get("precipitation_sum", [0])
                max_precip = max(precip) if precip else 0
                if max_precip > 50:
                    risk_factors["precipitation_risk"] = min(max_precip / 100, 1.0)
        
        # Elevation risk (prioritize open_elevation)
        elevation_sources = ["open_elevation", "opentopo"]
        elevation = 500  # default
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
        
        # Water proximity risk
        if water_bodies and "elements" in water_bodies:
            water_count = len(water_bodies["elements"])
            risk_factors["water_proximity_risk"] = min(water_count / 10, 1.0)
        
        # Overall risk calculation
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

# 🤖 AI Analysis with Gemini
# Fix Gemini JSON parsing in gemini_analyze_flood_risk
def gemini_analyze_flood_risk(all_data, risk_assessment):
    prompt = f"""
Flood Risk Analysis for Pakistan - Data Analysis:
WEATHER DATA (Precipitation Forecast): {json.dumps(all_data.get('weather', {}).get('open_meteo', {}).get('daily', {}).get('precipitation_sum', [0] * 7), indent=1)}
ELEVATION: {json.dumps(all_data.get('elevation', {}), indent=1)[:1000]}
WATER BODIES: {json.dumps(all_data.get('water_bodies', {}), indent=1)[:1000]}
CALCULATED RISK ASSESSMENT: {json.dumps(risk_assessment, indent=1)}
SOIL DATA: {json.dumps(all_data.get('soil', {}), indent=1)[:1000]}
Instructions:
- Base your analysis strictly on the provided data, especially the precipitation forecast.
- If precipitation is 0 mm for all days, risk should be LOW unless other factors (e.g., water proximity, low elevation) strongly indicate otherwise.
- Provide a comprehensive flood risk analysis including:
  1. **Risk Level**: HIGH/MEDIUM/LOW with reasoning consistent with the data
  2. **Key Factors**: Most important contributing factors
  3. **Timeline**: When is risk highest (next 24h, 48h, week)
  4. **Recommendations**: Specific actionable advice
  5. **Urdu Summary**: 2-3 lines in Urdu for local communication
  6. **Critical Areas**: Specific locations most at risk
- Format as structured JSON with these exact keys:
  - risk_level
  - key_factors (array)
  - timeline
  - recommendations (array)
  - urdu_summary
  - critical_areas (array)
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        response_text = response.text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7  # Fixed missing quote
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
            try:
                ai_result = json.loads(json_str)
                precip = all_data.get('weather', {}).get('open_meteo', {}).get('daily', {}).get('precipitation_sum', [0] * 7)
                max_precip = max(precip) if precip else 0
                if max_precip < 10 and ai_result["risk_level"] != "LOW":
                    logging.warning(f"AI risk level {ai_result['risk_level']} inconsistent with precipitation {max_precip} mm; overriding to LOW")
                    ai_result["risk_level"] = "LOW"
                    ai_result["key_factors"] = ["No significant precipitation forecasted", "Moderate elevation reduces risk"]
                    ai_result["timeline"] = "No significant risk in the next 7 days"
                return ai_result
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini JSON: {e}")
                return {"error": "Invalid JSON response", "raw_data": response_text}
        else:
            logging.warning("Gemini response not in JSON format")
            return {"analysis": response_text, "format": "text"}
    except Exception as e:
        logging.error(f"Gemini analysis failed: {e}")
        return {"error": str(e), "raw_data_available": True}

# -----------------------------
# 5️⃣ Main Data Collection Orchestrator
# -----------------------------
def collect_flood_data(bbox, target_area_name="Unknown"):
    """Main function to collect all flood-related data"""
    
    # Calculate center point
    lat = (bbox[0] + bbox[2]) / 2
    lon = (bbox[1] + bbox[3]) / 2
    
    logging.info(f"🌊 Starting flood data collection for {target_area_name}")
    
    # Initialize output structure
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
            "weather": {},
            "elevation": {},
            "water_bodies": {},
            "infrastructure": {},
            "satellite": {},
            "soil": {},
            "landuse": {},
            "river": {},
            "historical": {},
            "satellite_flood": {}
        },
        "risk_assessment": {},
        "ai_analysis": {},
        "visualization": {},
        "status": "processing"
    }
    
    # 1. Weather Data Collection
    logging.info("📡 Fetching weather data...")
    output["data_sources"]["weather"] = fetch_weather_data(lat, lon)
    
    # 2. Elevation Analysis  
    logging.info("🏔️ Fetching elevation data...")
    output["data_sources"]["elevation"] = fetch_elevation_data(lat, lon)
    
    # 3. Water Bodies
    logging.info("🌊 Fetching water bodies...")
    output["data_sources"]["water_bodies"] = fetch_water_bodies(bbox)
    
    # 4. Infrastructure
    logging.info("🛣️ Fetching infrastructure data...")
    output["data_sources"]["infrastructure"] = fetch_infrastructure(bbox)
    
    # 5. Satellite Data
    logging.info("🛰️ Fetching satellite data...")  # Fixed emoji
    output["data_sources"]["satellite"] = fetch_satellite_data(lat, lon)
    
    # 6. Soil & Land Data
    logging.info("🌱 Fetching soil and land data...")
    output["data_sources"]["soil"] = fetch_soil_data(lat, lon)
    output["data_sources"]["landuse"] = fetch_landuse_data(lat, lon)
    
    # 7. River Flow Data
    logging.info("🌊 Fetching river flow/discharge data...")
    output["data_sources"]["river"] = fetch_river_data(lat, lon)
    
    # 8. Historical Flood Data
    logging.info("📜 Fetching historical flood events...")
    output["data_sources"]["historical"] = fetch_historical_floods(lat, lon)
    
    # 9. Satellite Flood Extent (Placeholder)
    logging.info("🛰️ Fetching satellite flood extent...")
    output["data_sources"]["satellite_flood"] = {"status": "Placeholder - implement Sentinel Hub API"}
    
    # 10. Risk Assessment
    logging.info("📊 Calculating flood risk...")
    output["risk_assessment"] = calculate_flood_risk(
        output["data_sources"]["weather"],
        output["data_sources"]["elevation"],
        output["data_sources"]["water_bodies"]
    )
    
    # # 11. AI Analysis
    # logging.info("🤖 Running AI analysis...")
    # output["ai_analysis"] = gemini_analyze_flood_risk(output["data_sources"], output["risk_assessment"])
    
    # 12. Visualization (Precipitation Forecast)
    if "open_meteo" in output["data_sources"]["weather"]:
        precip = output["data_sources"]["weather"]["open_meteo"].get("daily", {}).get("precipitation_sum", [0] * 7)
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
    
    # 13. Final status
    output["status"] = "complete"
    output["meta"]["completion_time"] = int(time.time())
    
    # 14. Save output to JSON file
    output_file = f"flood_report_{target_area_name.replace(' ', '_')}_{output['meta']['timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.info(f"📝 Saved flood report to {output_file}")
    
    logging.info("✅ Flood data collection complete!")
    return output

# -----------------------------
# 6️⃣ Pakistan-specific presets
# -----------------------------
PAKISTAN_AREAS = {
    "karachi": {
        "bbox": [24.7, 66.9, 25.2, 67.5],
        "name": "Karachi Metropolitan"
    },
    "lahore": {
        "bbox": [31.4, 74.2, 31.7, 74.5],
        "name": "Lahore District"
    },
    "islamabad": {
        "bbox": [33.6, 72.8, 33.8, 73.2],
        "name": "Islamabad Capital Territory"
    },
    "faisalabad": {
        "bbox": [31.3, 73.0, 31.5, 73.2],
        "name": "Faisalabad District"
    },
    "rawalpindi": {
        "bbox": [33.5, 73.0, 33.7, 73.2],
        "name": "Rawalpindi District"
    },
    "multan": {
        "bbox": [30.1, 71.4, 30.3, 71.6],
        "name": "Multan District"
    },
    "peshawar": {
        "bbox": [34.0, 71.5, 34.2, 71.7],
        "name": "Peshawar District"
    },
    "quetta": {
        "bbox": [30.1, 66.9, 30.3, 67.1],
        "name": "Quetta District"
    }
}

# -----------------------------
# 7️⃣ Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\n🌍 Select target area for flood risk analysis:\n")
    for i, (key, val) in enumerate(PAKISTAN_AREAS.items(), 1):
        print(f"{i}. {val['name']} ({key})")
    print(f"{len(PAKISTAN_AREAS)+1}. Custom Area")

    choice = int(input("\nEnter choice number: "))

    if 1 <= choice <= len(PAKISTAN_AREAS):
        area_key = list(PAKISTAN_AREAS.keys())[choice-1]
        area_config = PAKISTAN_AREAS[area_key]
        bbox = area_config["bbox"]
        name = area_config["name"]
    else:
        print("\nEnter custom bounding box (min_lat, min_lon, max_lat, max_lon):")
        bbox = list(map(float, input("Format: lat1,lon1,lat2,lon2 → ").split(",")))
        name = "Custom Area"

    # Collect flood data
    flood_data = collect_flood_data(bbox, name)
    
    # # Print AI analysis results
    # print("\n📊 AI Analysis Results:")
    # print(json.dumps(flood_data["ai_analysis"], indent=2, ensure_ascii=False))
    
    # Print visualization configuration
    print("\n📈 Visualization Configuration (Chart.js):")
    print(json.dumps(flood_data["visualization"], indent=2))