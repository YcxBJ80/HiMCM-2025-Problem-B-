import requests
import csv
import pandas as pd
from haversine import haversine
import itertools
import os

###############################################
# 1. Download OpenFlights Airport Data
###############################################

def download_openflights_airports():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    print("Downloading OpenFlights airport data...")
    r = requests.get(url)
    r.raise_for_status()

    with open("airports.dat", "w", encoding="utf-8") as f:
        f.write(r.text)

    print("✓ Airport data saved to airports.dat")

if not os.path.exists("airports.dat"):
    download_openflights_airports()
else:
    print("airports.dat already exists, skipping download.")


###############################################
# 2. Define US State Capital Coordinates
###############################################

state_capitals_coords = {
    "Alabama": (32.3777, -86.3000),
    "Alaska": (58.3019, -134.4197),
    "Arizona": (33.4484, -112.0740),
    "Arkansas": (34.7465, -92.2896),
    "California": (38.5816, -121.4944),
    "Colorado": (39.7392, -104.9903),
    "Connecticut": (41.7658, -72.6734),
    "Delaware": (39.1582, -75.5244),
    "Florida": (30.4383, -84.2807),
    "Georgia": (33.7490, -84.3880),
    "Hawaii": (21.3099, -157.8581),
    "Idaho": (43.6150, -116.2023),
    "Illinois": (39.7983, -89.6544),
    "Indiana": (39.7684, -86.1581),
    "Iowa": (41.5868, -93.6250),
    "Kansas": (39.0473, -95.6752),
    "Kentucky": (38.2009, -84.8733),
    "Louisiana": (30.4515, -91.1871),
    "Maine": (44.3072, -69.7817),
    "Maryland": (38.9784, -76.4922),
    "Massachusetts": (42.3601, -71.0589),
    "Michigan": (42.7325, -84.5555),
    "Minnesota": (44.9537, -93.0900),
    "Mississippi": (32.2988, -90.1848),
    "Missouri": (38.5767, -92.1735),
    "Montana": (46.5891, -112.0391),
    "Nebraska": (40.8136, -96.7026),
    "Nevada": (39.1638, -119.7674),
    "New Hampshire": (43.2081, -71.5376),
    "New Jersey": (40.2206, -74.7597),
    "New Mexico": (35.6870, -105.9378),
    "New York": (42.6526, -73.7562),
    "North Carolina": (35.7796, -78.6382),
    "North Dakota": (46.8083, -100.7837),
    "Ohio": (39.9612, -82.9988),
    "Oklahoma": (35.4676, -97.5164),
    "Oregon": (44.9429, -123.0351),
    "Pennsylvania": (40.2732, -76.8867),
    "Rhode Island": (41.8236, -71.4222),
    "South Carolina": (34.0007, -81.0348),
    "South Dakota": (44.3683, -100.3509),
    "Tennessee": (36.1627, -86.7816),
    "Texas": (30.2672, -97.7431),
    "Utah": (40.7608, -111.8910),
    "Vermont": (44.2601, -72.5754),
    "Virginia": (37.5407, -77.4360),
    "Washington": (47.0379, -122.9007),
    "West Virginia": (38.3498, -81.6326),
    "Wisconsin": (43.0731, -89.4012),
    "Wyoming": (41.1400, -104.8202)
}


###############################################
# 3. Load Airport Data and Match Nearest Airport
###############################################

airports = pd.read_csv(
    "airports.dat",
    header=None,
    names=["ID", "Name", "City", "Country", "IATA", "ICAO",
           "Lat", "Lon", "Alt", "TZ", "DST", "TzDB", "Type", "Source"]
)

us_airports = airports[(airports["Country"] == "United States") & (airports["IATA"].notna())]

state_to_airport = {}

print("Matching each state capital to nearest airport...")

for state, coord in state_capitals_coords.items():
    min_dist = 1e9
    best_airport = None

    for _, row in us_airports.iterrows():
        ap_coord = (row["Lat"], row["Lon"])
        d = haversine(coord, ap_coord)
        if d < min_dist:
            min_dist = d
            best_airport = (row["IATA"], ap_coord)

    state_to_airport[state] = best_airport
    print(f"  {state} → {best_airport[0]} (distance {min_dist:.1f} km)")


###############################################
# 4. OSRM Driving Distance + Time
###############################################

def get_osrm_driving(origin, dest):
    lat1, lon1 = origin
    lat2, lon2 = dest
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

    try:
        res = requests.get(url).json()
        route = res["routes"][0]
        return route["distance"], route["duration"]
    except:
        return None, None


###############################################
# 5. Compute All State Pairs
###############################################

results = []

print("\nComputing all state pairs...\n")

for (stateA, coordA), (stateB, coordB) in itertools.combinations(state_capitals_coords.items(), 2):
    print(f"Processing {stateA} → {stateB}")

    # Driving data
    drive_dist, drive_time = get_osrm_driving(coordA, coordB)

    # Flight data
    IATA_A, airportA_coord = state_to_airport[stateA]
    IATA_B, airportB_coord = state_to_airport[stateB]

    flight_dist_km = haversine(airportA_coord, airportB_coord)
    flight_dist_m = flight_dist_km * 1000

    # Flight time estimate (850 km/h)
    flight_time_s = (flight_dist_km / 850.0) * 3600.0

    results.append([
        stateA, stateB,
        drive_dist, drive_time,
        flight_dist_m, flight_time_s
    ])


###############################################
# 6. Save CSV
###############################################

df = pd.DataFrame(results, columns=[
    "State_A", "State_B",
    "Driving_Distance_m", "Driving_Time_s",
    "Flight_Distance_m", "Flight_Time_s"
])

df.to_csv("state_transport_stats.csv", index=False)
print("\n✓ Saved results to state_transport_stats.csv")


###############################################
# 7. Compute Averages
###############################################

df["distance_ratio"] = df["Driving_Distance_m"] / df["Flight_Distance_m"]
df["time_ratio"] = df["Driving_Time_s"] / df["Flight_Time_s"]

print("\n===== Final Result =====")
print("Average Distance Ratio (Driving / Flying):", df["distance_ratio"].mean())
print("Average Time Ratio (Driving / Flying):", df["time_ratio"].mean())
print("========================")
