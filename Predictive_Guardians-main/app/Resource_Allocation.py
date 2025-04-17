import os

import h2o
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

# Determine the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def load_data_recidivism():
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Recidivism_cleaned_data.csv')
    return pd.read_csv(data_file_path)

@st.cache_data
def load_anomaly_feedback():
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'feedback_cleaned.csv')
    return pd.read_csv(data_file_path)

@st.cache_data
def load_resource_data():
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Resource_Data.csv')
    return pd.read_csv(data_file_path)

@st.cache_resource
def load_model_recidivism():
    return None  # Placeholder since we do not have the actual model

@st.cache_resource
def load_individual_models():
    try:
        model_paths = [
            os.path.join(root_dir, 'models', 'Recidivism_model', f'model_{i}.zip') for i in range(1, 6)
        ]
        missing = any(not os.path.exists(path) for path in model_paths)
        if missing:
            return [DummyClassifier(strategy="most_frequent") for _ in range(5)]

        individual_models = [h2o.import_mojo(path) for path in model_paths]
        return individual_models
    except:
        return [DummyClassifier(strategy="most_frequent") for _ in range(5)]

def get_unique_values(data, feature):
    return data[feature].unique().tolist()

def detect_anomalies(df):
    st.subheader("Feedback Anomaly Detection")
    st.write("Detecting anomalies in public feedback based on sentiment scores...")

    mean = df['SentimentScore'].mean()
    std = df['SentimentScore'].std()

    threshold = 2
    anomalies = df[(df['SentimentScore'] < mean - threshold * std) | (df['SentimentScore'] > mean + threshold * std)]

    if not anomalies.empty:
        st.warning("Unusual feedback sentiment detected in the following records:")
        st.dataframe(anomalies[['Date', 'District', 'Feedback', 'SentimentScore']])
    else:
        st.success("No unusual sentiment patterns detected.")

def forecast_resources(df):
    st.subheader("Resource Forecasting")
    st.write("Forecasting future police resource needs based on crime rates...")

    df_sorted = df.sort_values(by='Year')
    X = df_sorted[['Year']]
    y = df_sorted['Crime Count']

    model = LinearRegression()
    model.fit(X, y)

    next_year = df_sorted['Year'].max() + 1
    predicted_crime = model.predict([[next_year]])[0]

    st.write(f"Predicted crime count for {next_year}: **{int(predicted_crime)}**")

    recommended_asi = int(predicted_crime / 50)
    recommended_chc = int(predicted_crime / 40)
    recommended_cpc = int(predicted_crime / 20)

    st.write("### Recommended Additional Resources")
    st.write(f"ASI: {recommended_asi}, CHC: {recommended_chc}, CPC: {recommended_cpc}")

def resource_allocation(df):
    import numpy as np
    from pulp import LpVariable, LpProblem, LpMaximize, lpSum

    def optimise_resource_allocation(district_name, sanctioned_asi, sanctioned_chc, sanctioned_cpc):
        problem = LpProblem("Optimal_Resource_Allocation", LpMaximize)

        asi_vars = LpVariable.dicts("ASI", district_name.index, lowBound=0, cat='Integer')
        chc_vars = LpVariable.dicts("CHC", district_name.index, lowBound=0, cat='Integer')
        cpc_vars = LpVariable.dicts("CPC", district_name.index, lowBound=0, cat='Integer')

        problem += lpSum(district_name.loc[i, 'Normalised Crime Severity'] * (asi_vars[i] + chc_vars[i] + cpc_vars[i]) for i in district_name.index)

        problem += lpSum(asi_vars[i] for i in district_name.index) <= sanctioned_asi
        problem += lpSum(chc_vars[i] for i in district_name.index) <= sanctioned_chc
        problem += lpSum(cpc_vars[i] for i in district_name.index) <= sanctioned_cpc

        for i in district_name.index:
            problem += asi_vars[i] + chc_vars[i] + cpc_vars[i] >= 1

        for i in district_name.index:
            problem += asi_vars[i] <= max(1, sanctioned_asi * district_name.loc[i, 'Normalised Crime Severity'])
            problem += chc_vars[i] <= max(1, sanctioned_chc * district_name.loc[i, 'Normalised Crime Severity'])
            problem += cpc_vars[i] <= max(1, sanctioned_cpc * district_name.loc[i, 'Normalised Crime Severity'])

        st.write("Calculating crime severity based on crime types and crime frequency for allocating resources accordingly...")
        problem.solve()

        district_name['Allocated ASI'] = [asi_vars[i].varValue for i in district_name.index]
        district_name['Allocated CHC'] = [chc_vars[i].varValue for i in district_name.index]
        district_name['Allocated CPC'] = [cpc_vars[i].varValue for i in district_name.index]

        district_name[['Allocated ASI', 'Allocated CHC', 'Allocated CPC']] = district_name[['Allocated ASI', 'Allocated CHC', 'Allocated CPC']].apply(np.round).astype(int)

        return district_name

    def allocate_resources(option, district_name, updated_asi, updated_chc, updated_cpc):
        st.write(f"### Current sanctioned strengths for {option}:")
        st.write(f"ASI: {updated_asi}, CHC: {updated_chc}, CPC: {updated_cpc}")

        st.write("### Resource allocation in progress...")

        updated_district = optimise_resource_allocation(district_name, updated_asi, updated_chc, updated_cpc)

        st.write("### Allocation complete.")
        st.write("You can now view the resource allocation for specific police units.")

        police_units = ["All"] + list(district_name["Police Unit"].unique())
        selected_units = st.multiselect("Select Police Units to view allocation:", police_units)

        if "All" in selected_units:
            selected_data = updated_district
        else:
            selected_data = updated_district[updated_district["Police Unit"].isin(selected_units)]

        if st.button("Show Allocation"):
            selected_data = selected_data.reset_index(drop=True)
            st.table(selected_data[["Village Area Name", "Beat Name", "Normalised Crime Severity", "Allocated ASI", "Allocated CHC", "Allocated CPC"]])
            st.session_state.default = False
            st.session_state.apply = False

    st.title("Police Resource Allocation and Management")
    options = ["Select the District"] + list(df["District Name"].unique())
    option = st.selectbox("Select an option", options)

    if option != "Select the District":
        district_name = df[df["District Name"] == option]
        st.write(f"### Selected District: {option}")

        default_asi = int(district_name['Sanctioned Strength of Assistant Sub-Inspectors per District'].iloc[0])
        default_chc = int(district_name['Sanctioned Strength of Head Constables per District'].iloc[0])
        default_cpc = int(district_name['Sanctioned Strength of Police Constables per District'].iloc[0])

        sanctioned_asi = st.number_input("Sanctioned Assistant Sub-Inspectors [ASI]", value=default_asi, min_value=int(default_asi * 0.9), max_value=int(default_asi * 1.1), step=1)
        sanctioned_chc = st.number_input("Sanctioned Head Constables [CHC]", value=default_chc, min_value=int(default_chc * 0.9), max_value=int(default_chc * 1.1), step=1)
        sanctioned_cpc = st.number_input("Sanctioned Police Constables [CPC]", value=default_cpc, min_value=int(default_cpc * 0.9), max_value=int(default_cpc * 1.1), step=1)

        if "default" not in st.session_state:
            st.session_state.default = False

        if "apply" not in st.session_state:
            st.session_state.apply = False

        default = st.button("Use default sanctioned strengths")
        apply = st.button("Apply")

        if (default or st.session_state.default) and not st.session_state.apply:
            st.session_state.apply = False
            st.session_state.default = True
            allocate_resources(option, district_name, default_asi, default_chc, default_cpc)

        if (apply or st.session_state.apply) and not st.session_state.default:
            st.session_state.default = False
            st.session_state.apply = True
            allocate_resources(option, district_name, sanctioned_asi, sanctioned_chc, sanctioned_cpc)

if __name__ == '__main__':
    df = load_resource_data()
    resource_allocation(df)