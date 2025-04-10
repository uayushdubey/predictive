import requests

from Continuous_Learning_and_Feedback import *
from Crime_Pattern_Analysis import *
from Criminal_Profiling import create_criminal_profiling_dashboard
from Predictive_modeling import *
from Resource_Allocation import *

# Set root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Sidebar menu using emojis
with st.sidebar:
    st.markdown("## 🛡️ Predictive Guardians")
    selected = st.radio("Navigate", [
        '🏠 Home',
        '📊 Crime Pattern Analysis',
        '🧬 Criminal Profiling',
        '📈 Predictive Modeling',
        '🗺️ Police Resource Allocation and Management',
        '🔄 Continuous Learning and Feedback',
        '📚 Documentation and Resources'
    ])

# Helper to clean emoji prefix
selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected

# ------------------ Home ------------------
if selected_clean == "Home":
    st.title("Welcome to Predictive Guardians 🚔💻")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        Predictive Guardians is an innovative, AI-powered solution that revolutionizes the way law enforcement agencies approach public safety. By utilizing advanced data analysis and machine learning, my platform empowers agencies to make data-driven decisions, enabling them to allocate resources more efficiently and effectively.

        Predictive Guardians provides law enforcement agencies with the insights and actionable intelligence they need to stay one step ahead of criminals. My solution covers a comprehensive suite of analytical tools, including:

        - **Crime Pattern Analysis**: Uncover hidden insights and trends through spatial, temporal, and cluster-based analysis.
        - **Criminal Profiling**: Develop targeted crime prevention strategies by understanding the characteristics and behavioral patterns of offenders.
        - **Predictive Modeling**: Forecast future crime trends and patterns, enabling proactive resource allocation and intervention.
        - **Resource Allocation**: Optimize the deployment of police personnel to ensure efficient and effective utilization of law enforcement resources.
        - **Continuous Learning and Feedback**: Facilitate ongoing system improvement by incorporating user feedbacks, alerts, organizing collaborative learning sessions, and maintaining a knowledge base to document insights and lessons learned.

        Join me on this transformative journey as we redefine the future of public safety and ensure that our communities are safe, secure, and resilient. With Predictive Guardians, the path to a safer tomorrow is within reach.
        """)
        if st.button("Learn More"):
            st.session_state.selected_page = "Documentation and Resources"
            st.experimental_rerun()

    with col2:
        image_path = os.path.join(root_dir, 'assets', 'Home_Page_image.jpg')
        st.image(image_path, use_container_width=True)

# ------------------ Documentation ------------------

# ------------------ Crime Pattern Analysis ------------------
if selected_clean == "Crime Pattern Analysis":
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

    st.subheader("Temporal Analysis of Crime Data")
    temporal_analysis(crime_pattern_analysis)

    st.subheader("Choropleth Maps")
    chloropleth_maps(crime_pattern_analysis, geojson_data, mean_lat, mean_lon)

    st.subheader("Crime Hotspot Map")
    crime_pattern_analysis = crime_pattern_analysis.reset_index(drop=True)
    crime_pattern_analysis['Date'] = pd.to_datetime(crime_pattern_analysis[['Year', 'Month', 'Day']])
    mean_lat_sampled = crime_pattern_analysis['Latitude'].mean()
    mean_lon_sampled = crime_pattern_analysis['Longitude'].mean()
    crime_hotspots(crime_pattern_analysis, mean_lat_sampled, mean_lon_sampled)

# ------------------ Criminal Profiling ------------------
if selected_clean == "Criminal Profiling":
    create_criminal_profiling_dashboard()

# ------------------ Predictive Modeling ------------------
if selected_clean == "Predictive Modeling":
    predictive_modeling_recidivism()

# ------------------ Police Resource Allocation ------------------
if selected_clean == "Police Resource Allocation and Management":
    data_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Allocation_Cleaned.csv')
    df = pd.read_csv(data_path)
    resource_allocation(df)

# ------------------ Continuous Learning and Feedback ------------------
if selected_clean == "Continuous Learning and Feedback":
    continuous_learning_and_feedback()
