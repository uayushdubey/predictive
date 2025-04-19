import streamlit as st
import time
import os
import pandas as pd
import requests
from streamlit_extras.stylable_container import stylable_container

from Continuous_Learning_and_Feedback import *
from Crime_Pattern_Analysis import *
from Criminal_Profiling import create_criminal_profiling_dashboard
from Predictive_modeling import *
from Resource_Allocation import *

# Set root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --------------- Session State Initialization ----------------
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False

if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"

if 'tutorial_step' not in st.session_state:
    st.session_state.tutorial_step = 0  # Track tutorial step

# ------------------ Initial Welcome Screen ------------------
if not st.session_state.page_loaded:
    st.markdown("""
        <style>
            .fade-in {
                animation: fadeIn 2s ease-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
            .button:hover {
                background-color: #007bff;
                color: white;
                transition: 0.3s;
            }
        </style>
        <div style="height:100vh; display:flex; justify-content:center; align-items:center; flex-direction:column;" class="fade-in">
            <h1 style="font-size: 3em; color: #2c3e50; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);">🕵️‍♂️ Recidivism Risk Prediction</h1>
            <h3 style="font-size: 2em; color: #007bff;">Using Machine Learning and Demographic Profiling</h3>
            <p style="font-size: 1.2em; font-style: italic;">Predictive Guardians - AI-Powered Crime Intelligence and Resource Optimization Platform</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.page_loaded = True
    st.rerun()

# ------------------ Sidebar Menu ------------------
with st.sidebar:
    st.markdown("## 🛡️ Predictive Guardians")
    selected = st.radio("📌 Navigate to", [
        '🏠 Home',
        '📊 Crime Pattern Analysis',
        '🧬 Criminal Profiling',
        '📈 Predictive Modeling',
        '🗺️ Police Resource Allocation and Management',
        '🔄 Continuous Learning and Feedback',
    ])
    st.markdown("<hr style='margin-top:20px; border-color:#ccc;'>", unsafe_allow_html=True)

selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected

# ------------------ Home Page ------------------
if selected_clean == "Home":
    st.title("🚔 Welcome to Predictive Guardians")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(""" 
        <div style="text-align: justify; font-size: 1.1em;">
        <p><strong>Predictive Guardians</strong> is a next-gen platform empowering law enforcement with data-driven insights for proactive safety and smarter policing. With a Sherlock Holmes-inspired theme, this system leverages cutting-edge machine learning and data analytics techniques.</p>

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

        # Add button to start the tutorial
        if st.button("🚀 Explore the Tools", key="explore_tools", help="Click to begin your tutorial.", use_container_width=True):
            st.session_state.tutorial_step = 1
            st.session_state.selected_page = "Home"
            st.rerun()

    with col2:
        image_path = os.path.join(root_dir, 'assets', 'Home_Page_image.jpg')
        st.image(image_path, use_container_width=True)

# ------------------ Tutorial Steps ------------------

if 'tutorial_step' in st.session_state and st.session_state.tutorial_step > 0:
    if st.session_state.tutorial_step == 1:
        # Step 1: Introduction to Crime Pattern Analysis
        st.markdown("""
            <h2 style="color:#007bff;" class="fade-in">Step 1: Crime Pattern Analysis</h2>
            <p style="font-size:1.2em;">We are starting with the Crime Pattern Analysis feature. This tool provides temporal analysis, maps crime hotspots, and visualizes crime data on a choropleth map.</p>
            <p>Click "Next" to continue exploring this feature.</p>
        """, unsafe_allow_html=True)

        if st.button("Next", key="next1"):
            st.session_state.tutorial_step = 2
            st.rerun()

    elif st.session_state.tutorial_step == 2:
        # Step 2: Show Crime Pattern Analysis functionality
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

        if st.button("Next", key="next2"):
            st.session_state.tutorial_step = 3
            st.rerun()

    elif st.session_state.tutorial_step == 3:
        # Step 3: Introduction to Criminal Profiling
        st.markdown("""
            <h2 style="color:#007bff;" class="fade-in">Step 2: Criminal Profiling</h2>
            <p style="font-size:1.2em;">The Criminal Profiling feature helps identify patterns in criminal behavior, which can assist in anticipating future criminal activities and preventing crime.</p>
            <p>Click "Next" to explore Criminal Profiling.</p>
        """, unsafe_allow_html=True)

        if st.button("Next", key="next3"):
            st.session_state.tutorial_step = 4
            st.rerun()

    elif st.session_state.tutorial_step == 4:
        # Step 4: Show Criminal Profiling functionality
        create_criminal_profiling_dashboard()

        if st.button("Next", key="next4"):
            st.session_state.tutorial_step = 5
            st.rerun()

    elif st.session_state.tutorial_step == 5:
        # Step 5: Introduction to Predictive Modeling
        st.markdown("""
            <h2 style="color:#007bff;" class="fade-in">Step 3: Predictive Modeling</h2>
            <p style="font-size:1.2em;">This feature forecasts recidivism risk and potential criminal activity using machine learning models and demographic profiling.</p>
            <p>Click "Next" to continue.</p>
        """, unsafe_allow_html=True)

        if st.button("Next", key="next5"):
            st.session_state.tutorial_step = 6
            st.rerun()

    elif st.session_state.tutorial_step == 6:
        # Step 6: Show Predictive Modeling functionality
        predictive_modeling_recidivism()

        if st.button("Next", key="next6"):
            st.session_state.tutorial_step = 7
            st.rerun()

    elif st.session_state.tutorial_step == 7:
        # Step 7: Conclusion or next steps
        st.markdown("""
            <h2 style="color:#007bff;" class="fade-in">End of Tutorial</h2>
            <p style="font-size:1.2em;">You've completed the tour of the Predictive Guardians application. You can now explore the tools at your own pace.</p>
        """, unsafe_allow_html=True)

        if st.button("Finish", key="finish"):
            st.session_state.tutorial_step = 0
            st.session_state.selected_page = "Home"
            st.rerun()

# ------------------ Other Pages (e.g., Predictive Modeling, Resource Allocation) ------------------
if selected_clean == "Predictive Modeling":
    predictive_modeling_recidivism()

if selected_clean == "Police Resource Allocation and Management":
    data_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Allocation_Cleaned.csv')
    df = pd.read_csv(data_path)
    resource_allocation(df)

if selected_clean == "Continuous Learning and Feedback":
    continuous_learning_and_feedback()
