import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Determine the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def create_criminal_profiling_dashboard():
    # Construct the file path
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Criminal_Profiling_cleaned.csv')
    Criminal_Profiling = pd.read_csv(data_file_path)

    st.title("Criminal Profiling Dashboard")

    # Sidebar filtering options
    st.sidebar.markdown("### Filters")
    age_range = st.sidebar.slider("Select Age Range",
                                  min_value=Criminal_Profiling["age"].min(),
                                  max_value=Criminal_Profiling["age"].max(),
                                  value=(Criminal_Profiling["age"].min(), Criminal_Profiling["age"].max()))
    gender_filter = st.sidebar.multiselect("Select Gender", options=Criminal_Profiling['Sex'].unique(),
                                           default=Criminal_Profiling['Sex'].unique())
    occupation_filter = st.sidebar.multiselect("Select Occupation", options=Criminal_Profiling['Occupation'].unique(),
                                               default=Criminal_Profiling['Occupation'].unique())
    filtered_data = Criminal_Profiling[(Criminal_Profiling["age"] >= age_range[0]) &
                                       (Criminal_Profiling["age"] <= age_range[1]) &
                                       (Criminal_Profiling['Sex'].isin(gender_filter)) &
                                       (Criminal_Profiling['Occupation'].isin(occupation_filter))]

    # Age Distribution
    st.subheader("Age Distribution")
    fig = px.histogram(filtered_data, x="age", nbins=20, title="Age Distribution of Criminals",
                       color_discrete_sequence=['#FF5733'])
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        title_font_size=22,
        title_x=0.5,
        xaxis_title="Age",
        yaxis_title="Count",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        bargap=0.15
    )
    st.plotly_chart(fig)

    # Gender Analysis
    st.subheader("Gender Analysis")
    gender_counts = filtered_data['Sex'].value_counts()
    fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title="Gender Distribution",
                 color_discrete_sequence=['#2196F3', '#E91E63'])
    fig.update_traces(textinfo="percent+label", pull=[0.1, 0.1])
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        title_font_size=22,
        title_x=0.5,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    st.plotly_chart(fig)

    # Caste Analysis
    st.subheader("Caste Analysis")
    caste_counts = filtered_data[filtered_data['Caste'] != 'unknown']['Caste'].value_counts()
    fig = px.bar(x=caste_counts.index[:10], y=caste_counts.values[:10],
                 title="Top 10 Caste Distribution based on Crimes",
                 labels={'x': 'Caste', 'y': 'Count'},
                 color=caste_counts.values[:10], color_continuous_scale="YlOrRd")
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        title_font_size=22,
        title_x=0.5,
        xaxis_title="Caste",
        yaxis_title="Count",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='gray'),
        bargap=0.15
    )
    st.plotly_chart(fig)

    # Occupation Analysis
    st.subheader("Occupation Analysis")
    occupation_counts = \
    filtered_data[(filtered_data['Occupation'] != "unknown") & (filtered_data['Occupation'] != "Others PI Specify")][
        'Occupation'].value_counts()
    occupation_counts = occupation_counts.sort_values(ascending=False)[:10]  # Sort in descending order
    fig = px.bar(x=occupation_counts.values[::-1], y=occupation_counts.index[::-1], orientation='h',
                 title="Top 10 Occupation Associated with Criminal Activities",
                 labels={'x': 'Count', 'y': 'Occupation'},
                 color=occupation_counts.values[::-1], color_continuous_scale='Cividis')
    fig.update_layout(
        plot_bgcolor='#f5f5f5',
        title_font_size=22,
        title_x=0.5,
        xaxis_title="Count",
        yaxis_title="Occupation",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='gray'),
        margin=dict(t=30, b=30, l=50, r=30)
    )
    st.plotly_chart(fig)

    # Top Crime Categories and Sub-Categories
    st.subheader("Top Crime Categories and Sub-Categories")
    top_crime_groups = filtered_data['Crime_Group1'].value_counts().nlargest(5)
    top_crime_heads = filtered_data['Crime_Head2'].value_counts().nlargest(5)

    tabs = st.tabs(["Category", "Sub-Category"])

    with tabs[0]:
        fig = px.bar(x=top_crime_groups.index, y=top_crime_groups.values,
                     title="Top 5 Most Frequent Crime Group Categories",
                     labels={'x': 'Crime Group', 'y': 'Count'},
                     color=top_crime_groups.values, color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='#f5f5f5',
            title_font_size=22,
            title_x=0.5,
            xaxis_title="Crime Group",
            yaxis_title="Count",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            bargap=0.2
        )
        st.plotly_chart(fig)

    with tabs[1]:
        fig = px.bar(x=top_crime_heads.index, y=top_crime_heads.values, title="Top 5 Crime Groups Sub-Categories",
                     labels={'x': 'Crime Head', 'y': 'Count'},
                     color=top_crime_heads.values, color_continuous_scale='Purples')
        fig.update_layout(
            plot_bgcolor='#f5f5f5',
            title_font_size=22,
            title_x=0.5,
            xaxis_title="Crime Head",
            yaxis_title="Count",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            bargap=0.2
        )
        st.plotly_chart(fig)

    # Downloadable Data
    st.sidebar.download_button("Download Filtered Data (CSV)", data=filtered_data.to_csv(),
                               file_name="filtered_criminal_data.csv", mime="text/csv")

    # Explanation of Visuals
    st.markdown("""
        ### 🔎 How to Read This Dashboard:
        - **Age Distribution**: Shows the spread of ages of criminals involved in various crimes.
        - **Gender Distribution**: Visualizes the gender breakdown of criminal involvement.
        - **Caste Distribution**: Highlights the caste-related criminal patterns.
        - **Occupation Distribution**: Shows the top occupations associated with criminal behavior.
        - **Crime Categories**: Highlights the top crime categories and sub-categories.
        """, unsafe_allow_html=True)
