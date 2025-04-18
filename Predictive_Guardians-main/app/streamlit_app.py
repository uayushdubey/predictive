import streamlit as st
import pandas as pd
import os
import requests
import time
from streamlit_extras.stylable_container import stylable_container

# ---- Safe Module Imports with Fallback Errors ----
try:
    from Continuous_Learning_and_Feedback import *
except ModuleNotFoundError:
    st.error("⚠️ Missing module: Continuous_Learning_and_Feedback.py")

try:
    from Crime_Pattern_Analysis import *
except ModuleNotFoundError:
    st.error("⚠️ Missing module: Crime_Pattern_Analysis.py")

try:
    from Criminal_Profiling import create_criminal_profiling_dashboard
except ModuleNotFoundError:
    st.error("⚠️ Missing module: Criminal_Profiling.py")

try:
    from Predictive_modeling import *
except ModuleNotFoundError:
    st.error("⚠️ Missing module: Predictive_modeling.py")

try:
    from Resource_Allocation import *
except ModuleNotFoundError:
    st.error("⚠️ Missing module: Resource_Allocation.py")

# ---- Set root directory ----
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# ---- Session State Initialization ----
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False

if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"

# ---- Initial Welcome Screen ----
if not st.session_state.page_loaded:
    st.markdown("""
        <div style="height:100vh; display:flex; justify-content:center; align-items:center; flex-direction:column;">
            <h1 style="font-size: 3em; color: #2c3e50;">👮‍♂️ Welcome to <span style="color:#007bff;">Predictive Guardians</span></h1>
            <p style="font-size: 1.2em;">AI-Powered Crime Intelligence and Resource Optimization Platform</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.page_loaded = True
    st.rerun()

# ---- Sidebar Navigation ----
with st.sidebar:
    st.markdown("## 🛡️ Predictive Guardians")
    selected = st.radio("📌 Navigate to", [
        '🏠 Home',
        '📊 Crime Pattern Analysis',
        '🧬 Criminal Profiling',
        '📈 Predictive Modeling',
        '🗺️ Police Resource Allocation and Management',
        '🔄 Continuous Learning and Feedback',
        '📚 Documentation and Resources'
    ])
    st.markdown("<hr style='margin-top:20px; border-color:#ccc;'>", unsafe_allow_html=True)

selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected

# ---- Home Page ----
if selected_clean == "Home":
    st.title("🚔 Welcome to Predictive Guardians")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div style="text-align: justify; font-size: 1.1em;">
        <p><strong>Predictive Guardians</strong> is a next-gen platform empowering law enforcement with data-driven insights for proactive safety and smarter policing.</p>

        <p>Explore a comprehensive suite of intelligent tools:</p>
        <ul>
            <li><strong>Crime Pattern Analysis</strong>: Discover trends via spatial and temporal mapping.</li>
            <li><strong>Criminal Profiling</strong>: Uncover behavioral patterns to prevent crime more effectively.</li>
            <li><strong>Predictive Modeling</strong>: Forecast future crime occurrences for preemptive actions.</li>
            <li><strong>Resource Allocation</strong>: Strategize deployment of police units using AI recommendations.</li>
            <li><strong>Continuous Learning</strong>: Feedback loops, alerts, collaborative sessions & documentation.</li>
        </ul>

        <p>Let’s redefine public safety with <span style="color: #007bff;"><strong>AI and foresight</strong></span>.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Explore the Tools"):
            st.session_state.selected_page = "Crime Pattern Analysis"
            st.rerun()

    with col2:
        image_path = os.path.join(root_dir, 'assets', 'Home_Page_image.jpg')
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("📷 Home page image not found!")

# ---- Crime Pattern Analysis ----
if selected_clean == "Crime Pattern Analysis" and 'temporal_analysis' in globals():
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/adarshbiradar/maps-geojson/master/states/karnataka.json"
        response = requests.get(url)
        geojson_data = response.json()
        csv_path = os.path.join(root_dir, 'Component_datasets', 'Crime_Pattern_Analysis_Cleaned.csv')
        crime_data = pd.read_csv(csv_path)
        mean_lat = crime_data['Latitude'].mean()
        mean_lon = crime_data['Longitude'].mean()
        return mean_lat, mean_lon, geojson_data, crime_data

    mean_lat, mean_lon, geojson_data, crime_pattern_analysis = load_data()

    st.subheader("📅 Temporal Analysis of Crime Data")
    temporal_analysis(crime_pattern_analysis)

    st.subheader("🗺️ Choropleth Maps")
    chloropleth_maps(crime_pattern_analysis, geojson_data, mean_lat, mean_lon)

    st.subheader("🔥 Crime Hotspot Map")
    crime_pattern_analysis['Date'] = pd.to_datetime(crime_pattern_analysis[['Year', 'Month', 'Day']])
    mean_lat_sampled = crime_pattern_analysis['Latitude'].mean()
    mean_lon_sampled = crime_pattern_analysis['Longitude'].mean()
    crime_hotspots(crime_pattern_analysis, mean_lat_sampled, mean_lon_sampled)

# ---- Criminal Profiling ----
if selected_clean == "Criminal Profiling" and 'create_criminal_profiling_dashboard' in globals():
    create_criminal_profiling_dashboard()

# ---- Predictive Modeling ----
if selected_clean == "Predictive Modeling" and 'predictive_modeling_recidivism' in globals():
    predictive_modeling_recidivism()

# ---- Police Resource Allocation ----
if selected_clean == "Police Resource Allocation and Management" and 'resource_allocation' in globals():
    data_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Allocation_Cleaned.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        resource_allocation(df)
    else:
        st.error("❌ Resource allocation dataset not found!")

# ---- Continuous Learning and Feedback ----
if selected_clean == "Continuous Learning and Feedback" and 'continuous_learning_and_feedback' in globals():
    continuous_learning_and_feedback()
