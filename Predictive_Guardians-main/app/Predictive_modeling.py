import json
import os
import h2o
import joblib
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier

# Determine the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def load_data_recidivism():
    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Recidivism_cleaned_data.csv')
    return pd.read_csv(data_file_path)

# Load the model
@st.cache_resource
def load_model_recidivism():
    return None  # Placeholder since we do not have the actual model

# Fallback mock-up for individual models
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

# Get unique values for categorical features
def get_unique_values(data, feature):
    return data[feature].unique().tolist()

def predictive_modeling_recidivism():
    st.subheader("Repeat Offense Prediction App")
    st.write("Predict whether a previous accused person will commit a crime again.")

    try:
        h2o.init()
    except:
        st.warning("H2O initialization skipped (mock mode)")

    model = load_model_recidivism()

    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, '..'))

    scaler_path = os.path.join(project_root, 'models', 'Recidivism_model', 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    cleaned_data = load_data_recidivism()

    unique_castes = get_unique_values(cleaned_data, 'Caste')
    unique_professions = get_unique_values(cleaned_data, 'Profession')
    unique_districts = get_unique_values(cleaned_data, 'District_Name')
    unique_cities = get_unique_values(cleaned_data, 'PresentCity')

    with st.sidebar:
        st.markdown("### Provide input details for prediction")
        age = st.number_input("Age", min_value=7, max_value=100)
        caste = st.selectbox("Caste", unique_castes)
        profession = st.selectbox("Profession", unique_professions)
        present_district = st.selectbox("Crime District", unique_districts)
        present_city = st.selectbox("Criminal Present City", unique_cities)

    frequency_path = os.path.join(project_root, 'models', 'Recidivism_model', 'frequency_encoding.json')
    with open(frequency_path) as f:
        frequency = json.load(f)

    st.write("### Encoding Values")
    caste_encoded = frequency["Caste"].get(caste, 0)
    profession_encoded = frequency["Profession"].get(profession, 0)
    present_district_encoded = frequency["District_Name"].get(present_district, 0)
    present_city_encoded = frequency["PresentCity"].get(present_city, 0)

    st.write(f"Caste (encoded): {caste} → {caste_encoded}")
    st.write(f"Profession (encoded): {profession} → {profession_encoded}")
    st.write(f"District (encoded): {present_district} → {present_district_encoded}")
    st.write(f"City (encoded): {present_city} → {present_city_encoded}")

    new_data = pd.DataFrame({
        'District_Name': [present_district_encoded],
        'age': [age],
        'Caste': [caste_encoded],
        'Profession': [profession_encoded],
        'PresentCity': [present_city_encoded],
    })

    st.write("### Data Before Scaling")
    st.write(new_data)

    new_data_scaled = scaler.transform(new_data)
    new_df = pd.DataFrame(new_data_scaled, columns=new_data.columns, index=new_data.index)

    st.write("### Data After Scaling")
    st.write(new_df)

    try:
        new_dataframe = h2o.H2OFrame(new_df)
    except:
        new_dataframe = new_df

    st.markdown("""
    **Prediction Methodology:**
    - The model uses **Stacked Ensemble** (mocked for now) to predict the likelihood of a person re-offending.
    - The model considers features such as **age**, **caste**, **profession**, and **location**.
    - All predictions here are generated using placeholder models.
    """)

    individual_models = load_individual_models()
    model_names = ["K-Nearest Neighbors (KNN)", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting"]

    if st.button("Predict"):
        st.write("### Making Predictions...")

        predictions = []
        for model in individual_models:
            if isinstance(model, DummyClassifier):
                model.fit(new_data, [0] * len(new_data))
                predictions.append(model.predict(new_data).item())
            else:
                pred = model.predict(new_dataframe)
                predictions.append(int(pred.as_data_frame().loc[0, "predict"]))

        st.write("### Top 5 Model Predictions:")
        for i, (pred, name) in enumerate(zip(predictions, model_names)):
            result = 'Likely to repeat' if pred == 1 else 'Not likely to repeat'
            st.write(f"Model {i+1} ({name}): {result}")

        final_result = "🔴 The person is **likely** to repeat the crime." if sum(predictions) / len(predictions) > 0.5 else "🔵 The person is **not likely** to repeat the crime."

        st.markdown("### Final Outcome:")
        st.markdown(final_result)

    st.markdown("""
    **How This Works:**
    - The app predicts the likelihood of repeat offenses based on past behavior and demographic features.
    - Input your details and get predictions instantly.

    **Note:** This version uses mock models. Replace with real models when available.
    """, unsafe_allow_html=True)