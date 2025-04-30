# 🔐 Recidivism Risk Prediction Using Machine Learning and Demographic Profiling

🌐 **Live App**: [https://advancepredictions.streamlit.app/](https://advancepredictions.streamlit.app/)  
📁 **GitHub Repo**: [https://github.com/uayushdubey/Recidivism_Risk_Prediction_Using_Machine_Learning_and_Demographic_Profiling](https://github.com/uayushdubey/Recidivism_Risk_Prediction_Using_Machine_Learning_and_Demographic_Profiling)

Developed by **Ayush Dubey**, this project is a modular AI-powered dashboard that helps predict recidivism risk using machine learning and demographic profiling. It is designed to support data-driven decision-making in the criminal justice system by combining real-time feedback, crime pattern analysis, and behavioral insights.

---

## 🎯 Objective

To build an interpretable, data-driven system that forecasts the likelihood of repeat offenses (recidivism) based on demographic and historical data using machine learning models.

---

## 🚀 Key Features

- 🧠 **Recidivism Risk Prediction**  
  Trains and deploys ML models to identify high-risk individuals likely to reoffend.

- 👥 **Demographic & Behavioral Profiling**  
  Visualizes patterns across attributes like age, gender, prior convictions, etc.

- 🗺️ **Crime Pattern Analysis**  
  Detects spatial and temporal crime trends to support preventive measures.

- 🚓 **Resource Allocation Insights**  
  Provides strategic guidance on police resource distribution.

- 🔁 **Feedback & Continuous Learning**  
  Integrates a user feedback mechanism to enhance model accuracy over time.

---

## 🧰 Tech Stack

| Layer              | Tools Used                           |
|-------------------|---------------------------------------|
| **Frontend**       | Streamlit                            |
| **Backend/ML**     | Python, Pandas, Scikit-learn          |
| **Data Storage**   | CSV Files                             |
| **Visualization**  | Streamlit Charts, Matplotlib          |
| **Deployment**     | Streamlit Cloud, Docker (optional)    |
| **Dev Environment**| VS Code + DevContainers               |

---

## 🗂️ Project Structure


---

## ⚙️ Workflow

1. **Data Ingestion**: CSV datasets are loaded from `Component_datasets/`.
2. **Data Transformation**: Cleaned and preprocessed using pandas.
3. **Model Training & Prediction**: Classification models (e.g., Logistic Regression, Random Forest) trained for recidivism prediction.
4. **Visualization & Insights**: Streamlit-based interactive dashboard for real-time exploration.
5. **Feedback System**: Captures user feedback and integrates it into future model iterations.

---

## 🧪 Getting Started

### Python Installation

```bash
git clone https://github.com/uayushdubey/Recidivism_Risk_Prediction_Using_Machine-_Learning_and_Demographic_Profiling.git
cd Recidivism_Risk_Prediction_Using_Machine-_Learning_and_Demographic_Profiling
pip install -r requirements.txt
streamlit run Home/Home.py
