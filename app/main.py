import requests
import logging
import json
from datetime import datetime, timedelta
import time
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache for API responses
api_cache = {}
CACHE_TTL = 3600  # 1-hour TTL

def get_cached_response(cache_key):
    """Check if response is cached and not expired"""
    if cache_key in api_cache:
        timestamp, data = api_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None

def set_cached_response(cache_key, data):
    """Cache response with timestamp"""
    api_cache[cache_key] = (time.time(), data)

def safe_get(url, params=None, timeout=15, retries=3):
    """Safe HTTP GET with error handling and caching"""
    cache_key = (url, str(params))
    cached = get_cached_response(cache_key)
    if cached:
        logging.info(f"Using cached response for {url}")
        return cached

    headers = {'User-Agent': 'SoilDataTest/1.0'}
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

def fetch_soil_data(lat, lon, area_name="Unknown"):
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

    # Static soil type for Lahore (or any urban area)
    logging.info("🌱 Using default urban soil profile for soil type...")
    soil_data["soil_type"] = {
        "sand": 40,  # Percentage
        "silt": 30,
        "clay": 25,
        "organic_carbon_density": 0.5,  # g/kg
        "note": "Default urban soil profile (sand-heavy, moderate drainage) for Lahore"
    }

    # Prepare output
    output = {
        "meta": {
            "area_name": area_name,
            "center": [lat, lon],
            "timestamp": int(time.time()),
            "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "version": "1.0",
            "source": "Open-Meteo (moisture) + static soil profile"
        },
        "soil_data": soil_data,
        "status": "complete"
    }

    # Save to JSON file
    output_file = f"soil_report_{area_name.replace(' ', '_')}_{output['meta']['timestamp']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.info(f"📝 Saved soil report to {output_file}")

    return output

def main():
    # Lahore coordinates (center of Lahore District bbox from your script)
    lat = (31.4 + 31.7) / 2  # 31.55
    lon = (74.2 + 74.5) / 2  # 74.35
    area_name = "Lahore District"

    logging.info(f"🌍 Fetching soil data for {area_name}...")
    soil_data = fetch_soil_data(lat, lon, area_name)

    # Print results
    print("\n📊 Soil Data Results:")
    print(json.dumps(soil_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()