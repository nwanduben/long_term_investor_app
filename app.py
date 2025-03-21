import streamlit as st
import pandas as pd
import joblib

# Load the trained model, label encoders, and scaler
model = joblib.load("artifacts/best_model.pkl")
label_encoders = joblib.load("artifacts/label_encoders.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# Define numerical columns that were scaled during preprocessing
num_cols = ["balance", "duration", "campaign", "pdays", "previous"]

# Load feature order from processed dataset to ensure correct column order
processed_df = pd.read_csv("data/processed/processed_data.csv")
feature_order = processed_df.drop(columns=["deposit"]).columns.tolist()

# App Title
st.title("EliteBank Long-Term Investor Predictor")

st.write("Please enter client details:")

# --- User Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)

# Use the saved label encoder classes for each categorical feature
job_option = st.selectbox("Job", label_encoders["job"].classes_)
marital_option = st.selectbox("Marital Status", label_encoders["marital"].classes_)
education_option = st.selectbox("Education", label_encoders["education"].classes_)
default_option = st.selectbox("Default", label_encoders["default"].classes_)
balance = st.number_input("Account Balance", value=0)
housing_option = st.selectbox("Housing", label_encoders["housing"].classes_)
loan_option = st.selectbox("Loan", label_encoders["loan"].classes_)
contact_option = st.selectbox("Contact", label_encoders["contact"].classes_)
day = st.number_input("Day", min_value=1, max_value=31, value=15)
month_option = st.selectbox("Month", label_encoders["month"].classes_)
duration = st.number_input("Call Duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Campaign Contacts", min_value=0, value=1)
pdays = st.number_input("Pdays", value=-1)
previous = st.number_input("Previous Contacts", value=0)
poutcome_option = st.selectbox("Poutcome", label_encoders["poutcome"].classes_)

# --- Build Input Data ---
# Transform categorical inputs using the saved encoders
input_data = {
    "age": age,
    "job": label_encoders["job"].transform([job_option])[0],
    "marital": label_encoders["marital"].transform([marital_option])[0],
    "education": label_encoders["education"].transform([education_option])[0],
    "default": label_encoders["default"].transform([default_option])[0],
    "balance": balance,
    "housing": label_encoders["housing"].transform([housing_option])[0],
    "loan": label_encoders["loan"].transform([loan_option])[0],
    "contact": label_encoders["contact"].transform([contact_option])[0],
    "day": day,
    "month": label_encoders["month"].transform([month_option])[0],
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": label_encoders["poutcome"].transform([poutcome_option])[0],
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# --- Apply Scaling to Numerical Features ---
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Reorder Columns to Match Training Order ---
input_df = input_df[feature_order]

st.write("#### Processed Input Data")
st.dataframe(input_df)

# --- Prediction ---
if st.button("Predict Investor Type"):
    # Get probability for long-term class (class 1)
    prob_long = model.predict_proba(input_df)[0][1]
    
    # Set a decision threshold (e.g., 0.15)
    threshold = 0.15
    if prob_long > threshold:
        prediction = "Long-Term Investor"
        confidence = prob_long
    else:
        prediction = "Short-Term Investor"
        confidence = 1 - prob_long  # Confidence for short-term is 1 - prob_long
    
    st.write("### Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2%}")
