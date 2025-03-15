import streamlit as st
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn

# ✅ Load trained models
log_model = joblib.load("models/logistic_regression.pkl")
rf_model = joblib.load("models/random_forest.pkl")
dt_model = joblib.load("models/decision_tree.pkl")

# ✅ Load feature names used during training
training_data = pd.read_csv("data/processed/bank_marketing_cleaned.csv")

# 🚨 Drop unnecessary columns that were NOT used during training
training_data = training_data.drop(columns=["day", "month"], errors="ignore")

# ✅ Get the correct feature order
feature_columns = training_data.drop(columns=["deposit"]).columns  # Exclude target column

# ✅ Streamlit App Title
st.title("📈 Long-Term Investor Detection")
st.markdown("This app predicts **whether an investor is Long-Term or Short-Term** based on their financial behavior.")

# ✅ Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Bank Balance", min_value=-5000, max_value=100000, value=2000)
duration = st.number_input("Last Contact Duration (seconds)", min_value=1, max_value=5000, value=300)
campaign = st.number_input("Number of Contacts", min_value=1, max_value=50, value=2)
pdays = st.number_input("Days Since Last Contact (-1 if never contacted)", min_value=-1, max_value=500, value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=50, value=1)

# ✅ Categorical Inputs
default = st.selectbox("Has Credit in Default?", ["no", "yes"])
housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

job = st.selectbox("Job Type", ["blue-collar", "entrepreneur", "housemaid", "management", 
                                "retired", "self-employed", "services", "student", "technician", 
                                "unemployed", "unknown"])
marital = st.selectbox("Marital Status", ["single", "married"])
education = st.selectbox("Education Level", ["secondary", "tertiary", "unknown"])
contact = st.selectbox("Preferred Contact Method", ["telephone", "unknown"])
poutcome = st.selectbox("Previous Campaign Outcome", ["other", "success", "unknown"])

# ✅ Convert categorical inputs to binary
default = 1 if default == "yes" else 0
housing = 1 if housing == "yes" else 0
loan = 1 if loan == "yes" else 0

# ✅ Create DataFrame from user inputs
user_input = pd.DataFrame([[age, balance, duration, campaign, pdays, previous, default, housing, loan, 
                            job, marital, education, contact, poutcome]],
                          columns=['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'default', 
                                   'housing', 'loan', 'job', 'marital', 'education', 'contact', 'poutcome'])

# ✅ One-hot encode categorical variables to match training format
user_input = pd.get_dummies(user_input)

# 🚨 **Ensure user input matches model training features**
user_input = user_input.reindex(columns=feature_columns, fill_value=0)

# ✅ Model Selection
model_choice = st.selectbox("Choose Model for Prediction", ["Logistic Regression", "Random Forest", "Decision Tree"])

# ✅ Load Selected Model
if model_choice == "Logistic Regression":
    model = log_model
elif model_choice == "Random Forest":
    model = rf_model
else:
    model = dt_model

# 🚨 **Ensure user input order exactly matches model training**
user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

# ✅ Prediction
if st.button("Predict Investor Type"):
    prediction = model.predict(user_input)[0]
    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.success("✅ This investor is a **Long-Term Investor!**")
    else:
        st.error("❌ This investor is a **Short-Term Investor!** Consider diversifying for long-term gains.")

    # ✅ SHAP Feature Importance
    st.subheader("📊 SHAP (Feature Importance)")

 # 🚨 Convert user input DataFrame to NumPy float format before passing to SHAP
user_input_array = user_input.astype(float).to_numpy()

# ✅ SHAP Explainability
st.subheader("📊 SHAP (Feature Importance)")

# ✅ Choose the right SHAP Explainer based on the selected model
if model_choice == "Random Forest":
    explainer = shap.TreeExplainer(model)  # Use TreeExplainer for RF
    shap_values = explainer.shap_values(user_input_array)

    fig, ax = plt.subplots()
    
    # ✅ Ensure correct SHAP format for binary classification
    if isinstance(shap_values, list):  
        shap_values = shap_values[1]  # Select only class 1 (long-term investor)

    # ✅ Convert SHAP values to NumPy to avoid dtype issues
    shap_values = np.array(shap_values)

    shap.summary_plot(shap_values, user_input_array, feature_names=feature_columns, show=False)
    st.pyplot(fig)

elif model_choice == "Logistic Regression":
    explainer = shap.Explainer(model, user_input_array)  # Use SHAP Explainer
    shap_values = explainer(user_input_array)

    fig, ax = plt.subplots()
    
    # ✅ Ensure SHAP values are properly extracted
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    # ✅ Convert SHAP values to NumPy to prevent dtype issues
    shap_values = np.array(shap_values)

    shap.summary_plot(shap_values, user_input_array, feature_names=feature_columns, show=False)
    st.pyplot(fig)

elif model_choice == "Decision Tree":
    explainer = shap.TreeExplainer(model)  # Use TreeExplainer for DT
    shap_values = explainer.shap_values(user_input_array)

    fig, ax = plt.subplots()
    
    # ✅ Ensure correct SHAP format for binary classification
    if isinstance(shap_values, list):  
        shap_values = shap_values[1]  # Select only class 1 (long-term investor)

    # ✅ Convert SHAP values to NumPy to avoid dtype issues
    shap_values = np.array(shap_values)

    shap.summary_plot(shap_values, user_input_array, feature_names=feature_columns, show=False)
    st.pyplot(fig)


    # ✅ LIME Explanation
    st.subheader("🔍 LIME Explanation")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        np.array(training_data.drop(columns=["deposit"])),
        feature_names=feature_columns.tolist(),
        class_names=["Short-Term", "Long-Term"],
        mode="classification"
    )
    
    exp = lime_explainer.explain_instance(np.array(user_input.iloc[0]), model.predict_proba)
    st.pyplot(exp.as_pyplot_figure())

# ✅ MLflow Experiment Tracking
st.sidebar.header("📊 MLflow Experiment Tracking")
with st.sidebar:
    st.write("Tracking MLflow experiments:")
    st.write(f"Model: **{model_choice}**")
    st.write("Experiment: **Investor Prediction**")
