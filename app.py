import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing objects
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")  # OneHotEncoder used during training
scaler = joblib.load("scaler.pkl")    # StandardScaler used during training

st.set_page_config(page_title="AEFI Predictor", layout="centered")
st.title("💉 AEFI Prediction App")
st.write("Predict whether a person is likely to experience an Adverse Event Following Immunization (AEFI) based on demographic and vaccine information.")

# Sidebar inputs
st.sidebar.header("Input Information")

age = st.sidebar.slider("Age (years)", 0, 120, 30)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
vax_type = st.sidebar.selectbox("Vaccine Type", ["COVID19", "FLU", "MMR", "VARICELLA"])  # Add more as needed
dose_series = st.sidebar.selectbox("Dose Series", ["1", "2", "UNKNOWN"])
days_since_vax = st.sidebar.slider("Days Since Vaccination", 0, 365, 10)

# Interaction feature
vax_severe_interaction = f"{vax_type}_0"  # Assume not severe outcome for input

# Create DataFrame for input
input_df = pd.DataFrame({
    "AGE_YRS": [age],
    "TIME_SINCE_VAX": [days_since_vax],
    "SEX": [sex],
    "VAX_TYPE": [vax_type],
    "VAX_DOSE_SERIES": [dose_series],
    "VAX_SEVERE_INTERACTION": [vax_severe_interaction]
})

# One-hot encode categorical features
cat_features = ["SEX", "VAX_TYPE", "VAX_DOSE_SERIES", "VAX_SEVERE_INTERACTION"]
encoded = encoder.transform(input_df[cat_features])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_features))

# Scale numerical features
num_features = ["AGE_YRS", "TIME_SINCE_VAX"]
scaled = scaler.transform(input_df[num_features])
scaled_df = pd.DataFrame(scaled, columns=num_features)

# Final input for model
final_input = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Prediction
if st.button("Predict AEFI"):
    prediction = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ AEFI Likely (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No AEFI Expected (Probability: {prob:.2f})")

st.markdown("---")
st.markdown("Created with ❤️ using Streamlit | Model: Random Forest")
