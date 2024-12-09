import pandas as pd
import json
import folium
from folium.plugins import MarkerCluster


def load_data():
    df = pd.read_csv("../city_data/step_by_step.csv")
    df.fillna("MISSING", inplace=True)
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors="coerce")
    df['host_since'] = pd.to_datetime(df['host_since'], errors="coerce").dt.date
    df['host_location'] = pd.to_numeric(df['host_location'], errors="coerce")
    return df.copy()


graph = load_data()
with open('it.json') as f:
    italy_geojson = json.load(f)
cities_of_interest = ['bg', 'bo', 'fi', 'mi', 'na', 'rm', 've']
filtered_geojson = {
    "type": "FeatureCollection",
    "features": [feature for feature in italy_geojson['features'] if feature['properties'].get('name') in cities_of_interest]
}

neighbourhood_counts = graph.groupby(['df_city_location', 'neighbourhood_cleansed', 'latitude', 'longitude']).size().reset_index(name='count')
m = folium.Map(location=[41.8719, 12.5674], zoom_start=6)
folium.GeoJson(filtered_geojson, name="Italy").add_to(m)
marker_cluster = MarkerCluster().add_to(m)
for _, row in neighbourhood_counts.iterrows():
    city_abbr = row['df_city_location']
    neighbourhood = row['neighbourhood_cleansed']
    count = row['count']
    latitude = row['latitude']
    longitude = row['longitude']
    folium.CircleMarker(
        location=[latitude, longitude],
        radius=count * 0.02,
        popup=f"{city_abbr.upper()} - {neighbourhood}: {count}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(marker_cluster)

m.save('data/mappings/NEW_italy_map.html')
