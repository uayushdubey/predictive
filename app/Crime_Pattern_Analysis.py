import branca.colormap as cm
import folium
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN

def temporal_analysis(crime_pattern_analysis):
    with st.container():
        st.markdown("### 📅 Temporal Crime Analysis", unsafe_allow_html=True)
        st.markdown("<p style='color:gray;'>Filter crimes by District, Group, and Time Granularity.</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            district_options = ["All Districts"] + sorted(crime_pattern_analysis["District_Name"].unique())
            selected_districts = st.multiselect("🏙️ Select District(s)", district_options, default=[])

        with col2:
            crime_group_options = ["All Crime Groups"] + sorted(crime_pattern_analysis["CrimeGroup_Name"].unique())
            selected_crime_groups = st.multiselect("🚨 Select Crime Group(s)", crime_group_options, default=[])

        with col3:
            selected_time_granularity = st.radio("🕒 Granularity", ["Year", "Month", "Day"], horizontal=True)

        filtered_df = crime_pattern_analysis.copy()
        if "All Districts" not in selected_districts and selected_districts:
            filtered_df = filtered_df[filtered_df["District_Name"].isin(selected_districts)]

        if "All Crime Groups" not in selected_crime_groups and selected_crime_groups:
            filtered_df = filtered_df[filtered_df["CrimeGroup_Name"].isin(selected_crime_groups)]

        if filtered_df.empty:
            st.warning("⚠️ No data available for the selected filters.")
        else:
            group_by_cols = {
                "Year": "Year",
                "Month": "Month",
                "Day": "Day"
            }
            group_col = group_by_cols[selected_time_granularity]
            data = filtered_df.groupby([group_col, "District_Name", "CrimeGroup_Name"]).size().reset_index(name="Count")

            fig = px.bar(
                data, x=group_col, y="Count", color="District_Name", barmode="group",
                hover_data=["CrimeGroup_Name"],
                title=f"{selected_time_granularity}-wise Crime Count"
            )
            fig.update_layout(xaxis_title=selected_time_granularity, yaxis_title="Count", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def crime_hotspot_analysis(df, mean_lat, mean_lon):
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=7)
    colormap = cm.LinearColormap(colors=['blue', 'yellow', 'red'], vmin=0, vmax=df['Count'].max())

    HeatMap(df[['Latitude', 'Longitude', 'Count']].values.tolist(),
            gradient={"0.4": 'blue', "0.65": 'yellow', "1.0": 'red'},
            radius=15).add_to(m)

    coords = df[['Latitude', 'Longitude']].values
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    df['Cluster'] = dbscan.fit_predict(coords)

    for cluster in df['Cluster'].unique():
        if cluster != -1:
            cluster_points = df[df['Cluster'] == cluster]
            center_lat = cluster_points['Latitude'].mean()
            center_lon = cluster_points['Longitude'].mean()
            count = cluster_points['Count'].sum()
            folium.Marker(
                [center_lat, center_lon],
                popup=f'Cluster {cluster}<br>Crimes: {count}',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

    colormap.add_to(m)
    colormap.caption = 'Crime Density'
    return m

def crime_hotspots(crime_pattern_analysis, mean_lat, mean_lon):
    st.markdown("### 🔥 Crime Hotspot Map")
    st.markdown("<p style='color:gray;'>Visualize crime clusters and heatmap by date and type.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 2])

    with col1:
        dates = st.radio("📆 Date Filter", ["All", "Custom Date Range"], horizontal=True)
        if dates == "All":
            date_range = (crime_pattern_analysis['Date'].min(), crime_pattern_analysis['Date'].max())
        else:
            date_range = st.date_input("Select date range",
                                       [crime_pattern_analysis['Date'].min(), crime_pattern_analysis['Date'].max()],
                                       key='date_range')
            if len(date_range) != 2:
                st.stop()

    with col2:
        crime_types = st.multiselect("🔍 Crime Group(s)", crime_pattern_analysis['CrimeGroup_Name'].unique())

    filtered_data = crime_pattern_analysis[
        (crime_pattern_analysis['Date'] >= pd.Timestamp(date_range[0])) &
        (crime_pattern_analysis['Date'] <= pd.Timestamp(date_range[1]))
    ]

    if crime_types:
        filtered_data = filtered_data[filtered_data['CrimeGroup_Name'].isin(crime_types)]

    if st.button("🔎 Show Hotspots") and not filtered_data.empty:
        aggregated_data = filtered_data.groupby(['District_Name', 'UnitName', 'Latitude', 'Longitude', 'CrimeGroup_Name'])\
                                       .size().reset_index(name='Count')

        mean_lat = aggregated_data['Latitude'].mean()
        mean_lon = aggregated_data['Longitude'].mean()

        m = crime_hotspot_analysis(aggregated_data, mean_lat, mean_lon)
        st.components.v1.html(m._repr_html_(), height=600)

        st.markdown("""<hr/>
        ✅ <strong>How to read this map:</strong><br>
        - Red zones = High density of crimes<br>
        - Markers = Cluster centers<br>
        - Filter the map by date and crime type above
        """, unsafe_allow_html=True)

def chloropleth_maps(df, geojson_data, mean_lat, mean_lon):
    st.markdown("### 🗏️ Choropleth Map Analysis")
    st.markdown("<p style='color:gray;'>Choose a metric to visualize district-wise crime impact.</p>", unsafe_allow_html=True)

    district_stats = df.groupby('District_Name').agg({
        'FIRNo': 'count',
        'VICTIM COUNT': 'sum',
        'Accused Count': 'sum'
    }).reset_index()

    selected_stat = st.selectbox('📊 Crime Metric', ['Crime Incidents', 'Total Victim Count', 'Total Accused Count'])

    color_column_map = {
        'Crime Incidents': ('FIRNo', 'Crime Incidents'),
        'Total Victim Count': ('VICTIM COUNT', 'Victim Count'),
        'Total Accused Count': ('Accused Count', 'Accused Count')
    }

    color_col, label = color_column_map[selected_stat]

    fig = px.choropleth_mapbox(
        district_stats,
        geojson=geojson_data,
        locations='District_Name',
        featureidkey="properties.district",
        color=color_col,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": mean_lat, "lon": mean_lon},
        opacity=0.6,
        labels={color_col: label},
        title=f"{selected_stat} by District"
    )

    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
