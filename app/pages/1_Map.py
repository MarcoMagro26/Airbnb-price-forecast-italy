import streamlit as st
import pandas as pd
import random
import json
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
from app.custom.custom_functions import random_color_generator


random.seed(874631)

st.set_page_config(
    page_title="Map",
    layout='wide'
)
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)
st.title("Listings GeoMap")

#@st.cache_data
#def load_data():
#    df = pd.read_csv("data/city_data/step_by_step.csv")
#    df.fillna("MISSING", inplace=True)
#    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
#    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'], errors='coerce')
#    df['price'] = pd.to_numeric(df['price'], errors="coerce")
#    df['host_since'] = pd.to_datetime(df['host_since'], errors="coerce").dt.date
#    df['host_location'] = pd.to_numeric(df['host_location'], errors="coerce")
#    return df.copy()
#
#
#graph = load_data()
#zona_options = sorted(graph.loc[:, 'neighbourhood_cleansed'].unique().tolist())
#zona = st.sidebar.multiselect('Select neighbourhoods:', zona_options + ["All"], default="All")
#if "All" in zona:
#    zona = zona_options
#
#response_time_list = sorted(graph['host_response_time'].unique().tolist())
#response_time = st.sidebar.multiselect('Select response time:', response_time_list + ["All"], default="All")
#if "All" in response_time:
#    response_time = response_time_list
#
#response_rate = st.sidebar.slider("Select host response rate", min_value=0, max_value=100, value=100)
#if graph['host_response_rate'].isnull().sum()>0:
#    response_rate_MISSING_box = st.toggle("Include NAs for Host Response Rate variable in visualization", value=False)
#
#acceptance_rate = st.sidebar.slider("Select host acceptance rate", min_value=0, max_value=100, value=100)
#if graph['host_acceptance_rate'].isnull().sum()>0:
#    acceptance_rate_MISSING_box = st.toggle("Include NAs for Host Acceptance Rate variable in visualization", value=False)
#
#price_range = st.sidebar.slider("Select price range",
#                                min_value=0.0,
#                                max_value=max(graph['price']),
#                                value=(0.0, max(graph['price'])),
#                                step=0.01)
#if graph['price'].isnull().sum()>0:
#    price_range_MISSING_box = st.toggle("Include NAs for Price variable in visualization", value=False)
#
#host_since = st.sidebar.date_input("Host since", min(graph['host_since']))
#
#host_distance = st.sidebar.slider("Distance between host house and listing location (0 km to 100+ km)",
#                                  min_value=min(graph['host_location']),
#                                  max_value=100.0,
#                                  value=(0.0, 20.0),
#                                  step=0.01)
#if host_distance == 100.0:
#    host_distance = max(graph['host_location'])
#if graph['host_location'].isnull().sum()>0:
#    host_distance_MISSING_box = st.toggle("Include NAs for host distance variable in visualization", value=False)
#
#superhost_list = sorted(graph['host_is_superhost'].unique().tolist())
#superhost = st.sidebar.multiselect('Is host a superhost?:', superhost_list + ["All"], default="All")
#if "All" in superhost:
#    superhost = superhost_list
#
#is_in_neighbourhood = graph['neighbourhood_cleansed'].isin(zona)
#is_in_response_time = graph['host_response_time'].isin(response_time)
#is_response_rate_valid = (graph['host_response_rate'] <= response_rate) | response_rate_MISSING_box
#is_acceptance_rate_valid = (graph['host_acceptance_rate'] <= acceptance_rate) | acceptance_rate_MISSING_box
#is_in_price_range = (price_range[0] < graph['price']) & (graph['price'] < price_range[1]) | price_range_MISSING_box
#is_host_since = graph['host_since'] >= host_since
#is_host_distance = (host_distance[0] < graph['host_location']) & (graph['host_location'] < host_distance[1]) | host_distance_MISSING_box
#is_host_superhost = graph['host_is_superhost'].isin(superhost)
#
#filtered_graph = graph[
#    is_in_neighbourhood &
#    is_in_response_time &
#    is_response_rate_valid &
#    is_acceptance_rate_valid &
#    is_in_price_range &
#    is_host_since &
#    is_host_distance &
#    is_host_superhost
#    ]
#
#st.metric("Selected listings:", value=f"{filtered_graph.shape[0]}/{graph.shape[0]}")

#st.map(filtered_graph,
#       size=3,
#       use_container_width=False
#)


#@st.cache_data
#def load_data():
#    df = pd.read_csv("data/city_data/step_by_step.csv")
#    df.fillna("MISSING", inplace=True)
#    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
#    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'], errors='coerce')
#    df['price'] = pd.to_numeric(df['price'], errors="coerce")
#    df['host_since'] = pd.to_datetime(df['host_since'], errors="coerce").dt.date
#    df['host_location'] = pd.to_numeric(df['host_location'], errors="coerce")
#    return df.copy()
#
#
#graph = load_data()
#with open('data/mappings/it.json') as f:
#    italy_geojson = json.load(f)
#cities_of_interest = ['bg', 'bo', 'fi', 'mi', 'na', 'rm', 've']
#filtered_geojson = {
#    "type": "FeatureCollection",
#    "features": [feature for feature in italy_geojson['features'] if feature['properties'].get('name') in cities_of_interest]
#}
#
#neighbourhood_counts = graph.groupby(['df_city_location', 'neighbourhood_cleansed', 'latitude', 'longitude']).size().reset_index(name='count')
#m = folium.Map(location=[41.8719, 12.5674], zoom_start=6)
#folium.GeoJson(filtered_geojson, name="Italy").add_to(m)
#marker_cluster = MarkerCluster().add_to(m)
#for _, row in neighbourhood_counts.iterrows():
#    city_abbr = row['df_city_location']
#    neighbourhood = row['neighbourhood_cleansed']
#    count = row['count']
#    latitude = row['latitude']
#    longitude = row['longitude']
#    folium.CircleMarker(
#        location=[latitude, longitude],
#        radius=count * 0.02,
#        popup=f"{city_abbr.upper()} - {neighbourhood}: {count}",
#        color='blue',
#        fill=True,
#        fill_color='blue'
#    ).add_to(marker_cluster)
#
#st_map = st_folium(m, width=725)


@st.cache_data
def load_data():
    with open("data/mappings/NEW_italy_map.html",'r') as f:
        html_data = f.read()
    return html_data

html_data = load_data()
components.html(html_data, width=1000, height=600)
