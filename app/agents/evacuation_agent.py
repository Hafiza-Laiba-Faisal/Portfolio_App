import requests
import json
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime

# Free APIs Configuration
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ORS_BASE_URL = "https://api.openrouteservice.org/v2"
ORS_API_KEY = "YOUR_ORS_API_KEY"  # Replace with your free ORS key

# Local storage file for camps
CAMPS_FILE = "pakistan_relief_camps.json"

# Dummy PDMA/NDMA Relief Camps Data (Pakistan-specific)
DUMMY_CAMPS = [
    {
        "name": "PDMA Karachi Relief Camp",
        "lat": 24.8607,
        "lon": 67.0011,
        "city": "Karachi",
        "capacity": 500,
        "contact": "021-12345678",
        "operator": "PDMA Sindh"
    },
    {
        "name": "NDMA Lahore Shelter",
        "lat": 31.5204,
        "lon": 74.3587,
        "city": "Lahore",
        "capacity": 300,
        "contact": "042-98765432",
        "operator": "NDMA Punjab"
    },
    {
        "name": "PDMA Islamabad Safe Haven",
        "lat": 33.6844,
        "lon": 73.0479,
        "city": "Islamabad",
        "capacity": 200,
        "contact": "051-111222333",
        "operator": "PDMA ICT"
    },
    {
        "name": "PDMA Peshawar Camp",
        "lat": 34.0151,
        "lon": 71.5249,
        "city": "Peshawar",
        "capacity": 400,
        "contact": "091-5556677",
        "operator": "PDMA KPK"
    }
]

class EvacuationAgent:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="pak_evac_agent")
        self.stored_camps = self.load_camps()
        # Add dummy camps if file is empty
        if not self.stored_camps:
            self.stored_camps = DUMMY_CAMPS
            self.save_camps(self.stored_camps)
    
    def load_camps(self):
        """Load stored camps from JSON file."""
        try:
            with open(CAMPS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_camps(self, camps):
        """Save camps details to JSON file."""
        with open(CAMPS_FILE, 'w') as f:
            json.dump(camps, f, indent=4)
        print("Pakistan relief camps details saved locally!")
    
    def get_coordinates(self, address):
        """Convert address to lat/lon using Nominatim (free)."""
        params = {
            'q': address + ", Pakistan",  # Pakistan-specific
            'format': 'json',
            'limit': 1
        }
        response = requests.get(NOMINATIM_URL, params=params)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])
        return None, None
    
    def find_nearest_camps(self, lat, lon, radius_km=20):
        """Find nearest relief camps (dummy data + stored)."""
        camps = []
        for camp in self.stored_camps:
            distance = geodesic((lat, lon), (camp['lat'], camp['lon'])).km
            if distance <= radius_km:
                camp_copy = camp.copy()
                camp_copy['distance_km'] = round(distance, 2)
                camps.append(camp_copy)
        
        # Sort by distance
        camps.sort(key=lambda x: x['distance_km'])
        return camps[:3]  # Top 3 nearest
    
    def get_safe_route(self, start_lat, start_lon, end_lat, end_lon):
        """Get route using ORS Directions API (free tier)."""
        coords = [[start_lon, start_lat], [end_lon, end_lat]]  # ORS uses [lon, lat]
        headers = {'Authorization': ORS_API_KEY}
        body = {
            'coordinates': coords,
            'profile': 'foot-walking',  # Walking route for accessibility
            'format': 'json'
        }
        response = requests.post(f"{ORS_BASE_URL}/directions/foot-walking/geojson", json=body, headers=headers)
        if response.status_code == 200:
            data = response.json()
            route = data['features'][0]['properties']
            distance = route['summary']['distance'] / 1000  # km
            duration = route['summary']['duration'] / 60  # minutes
            steps = []
            for segment in route['segments']:
                for step in segment['steps']:
                    steps.append(step['instruction'])
            return {
                'distance_km': round(distance, 2),
                'duration_min': round(duration, 1),
                'steps': steps[:5]  # First 5 steps
            }
        return None
    
    def check_hazards(self, lat, lon):
        """Check weather alerts using NWS API (placeholder for Pakistan)."""
        # Note: NWS is US-specific; replace with Pakistan Met Dept API if available
        url = f"https://api.weather.gov/points/{lat},{lon}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            forecast_url = data['properties']['forecast']
            forecast_resp = requests.get(forecast_url)
            if forecast_resp.status_code == 200:
                periods = forecast_resp.json()['properties']['periods']
                for period in periods[:2]:
                    if any(hazard in period['detailedForecast'].lower() for hazard in ['flood', 'monsoon', 'landslide']):
                        return "Warning: Potential hazards (flood/monsoon) detected. Avoid low-lying areas."
        return "No major hazards detected. Check Pakistan Met Dept for updates."
    
    def guide_user(self, user_location):
        """Main guide function for Pakistan."""
        print(f"Emergency Evacuation Agent (Pakistan) Activated! [{datetime.now().strftime('%Y-%m-%d %I:%M %p PKT')}]")
        
        # Get coordinates
        if isinstance(user_location, str):
            lat, lon = self.get_coordinates(user_location)
        else:
            lat, lon = user_location
        if not lat:
            print("Location not found. Try again (e.g., 'Karachi, Pakistan').")
            return
        
        print(f"Your location: {lat}, {lon}")
        
        # Check hazards
        hazard_msg = self.check_hazards(lat, lon)
        print(hazard_msg)
        
        # Find nearest camps
        camps = self.find_nearest_camps(lat, lon)
        if not camps:
            print("No nearby camps found. Contact NDMA: 051-111222333 or PDMA.")
            return
        
        print("\nNearest PDMA/NDMA Relief Camps:")
        for i, camp in enumerate(camps, 1):
            print(f"{i}. {camp['name']} - {camp['distance_km']} km away")
            print(f"   City: {camp['city']}")
            print(f"   Coordinates: {camp['lat']}, {camp['lon']}")
            print(f"   Capacity: {camp['capacity']} people")
            print(f"   Contact: {camp['contact']}")
            print(f"   Operator: {camp['operator']}")
        
        # Guide to nearest camp
        nearest = camps[0]
        print(f"\nGuiding to nearest camp: {nearest['name']} ({nearest['city']})")
        
        route = self.get_safe_route(lat, lon, nearest['lat'], nearest['lon'])
        if route:
            print(f"Route Distance: {route['distance_km']} km")
            print(f"Estimated Time: {route['duration_min']} minutes (walking)")
            print("Step-by-Step Directions:")
            for step in route['steps']:
                print(f"- {step}")
        else:
            print("Route calculation failed. Use NDMA app or Google Maps as backup.")
        
        print("\nStay safe! Follow NDMA/PDMA instructions. Emergency: 112 or 15")

# Usage Example
if __name__ == "__main__":
    agent = EvacuationAgent()
    
    # Example: Use address or (lat, lon)
    user_loc = "Karachi, Pakistan"  # Or (24.8607, 67.0011)
    # user_loc = (24.8607, 67.0011)
    
    agent.guide_user(user_loc)
    
    # View stored camps
    print("\nStored PDMA/NDMA Camps:")
    for camp in agent.stored_camps:
        print(f"- {camp['name']} at {camp['city']} ({camp['lat']}, {camp['lon']})")