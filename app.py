import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model and supporting files
# -----------------------------
with open("graduate_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Graduate Placement Prediction", page_icon="🎓")

st.title("🎓 Graduate Placement Prediction System")

st.write(
    "Enter the graduate's information below to predict the likelihood of employment."
)

# -----------------------------
# User Inputs
# -----------------------------

gender = st.selectbox("Gender", list(encoders["Gender"].classes_))

nationality = st.selectbox(
    "Nationality",
    list(encoders["Nationality"].classes_)
)

refugee = st.selectbox(
    "Refugee Status",
    list(encoders["Refugee Status"].classes_)
)

disability = st.selectbox(
    "Disability Status",
    list(encoders["Disability status"].classes_)
)

youth = st.selectbox(
    "Youth (18-35)",
    list(encoders["Youth (18-35)"].classes_)
)

program = st.selectbox(
    "Program Name",
    list(encoders["Program Name"].classes_)
)

sponsorship = st.selectbox(
    "Sponsorship Type",
    list(encoders["Sponsorship Type"].classes_)
)

cohort = st.selectbox(
    "Cohort",
    list(encoders["Cohort"].classes_)
)

education = st.selectbox(
    "Education Level",
    list(encoders["Education Level"].classes_)
)

grad_year = st.number_input(
    "Graduation Year",
    min_value=2018,
    max_value=2035,
    value=2025
)

grad_month = st.slider(
    "Graduation Month",
    1,
    12,
    6
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Employment"):

    input_data = pd.DataFrame([{
        "Gender": encoders["Gender"].transform([gender])[0],
        "Nationality": encoders["Nationality"].transform([nationality])[0],
        "Refugee Status": encoders["Refugee Status"].transform([refugee])[0],
        "Disability status": encoders["Disability status"].transform([disability])[0],
        "Youth (18-35)": encoders["Youth (18-35)"].transform([youth])[0],
        "Program Name": encoders["Program Name"].transform([program])[0],
        "Sponsorship Type": encoders["Sponsorship Type"].transform([sponsorship])[0],
        "Cohort": encoders["Cohort"].transform([cohort])[0],
        "Education Level": encoders["Education Level"].transform([education])[0],
        "Graduation Year": grad_year,
        "Graduation Month": grad_month
    }])

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ This graduate is predicted to be EMPLOYED.")
    else:
        st.error("❌ This graduate is predicted to be NOT EMPLOYED.")