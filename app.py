import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Graduate Placement Predictor", layout="wide")

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open("models/graduate_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model_and_encoders()

# Title
st.title("🎓 Graduate Placement Predictor")
st.markdown("**Refactory Academy** — Predict employment outcomes for recent graduates")

# Sidebar: Instructions
with st.sidebar:
    st.markdown("### How It Works")
    st.info(
        """
        Enter a graduate's information below to predict their likelihood of employment.
        
        **Model Performance:** 77% accuracy on test data
        - Employed graduates: 79% precision
        - Job-seeking graduates: 73% precision
        """
    )

# Main form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Graduate Information")
    
    program = st.selectbox(
        "Training Program",
        options=["Program 1", "Program 2", "Program 3", "Program 4"],
        help="Which program did the graduate complete?"
    )
    
    education = st.selectbox(
        "Education Level",
        options=["Diploma", "Bachelor's", "Master's", "Unknown"],
        help="Highest level of education"
    )
    
    sponsorship = st.selectbox(
        "Sponsorship Type",
        options=["Fully Sponsored", "Partially Sponsored", "Self-Sponsored"],
        help="Who funded their training?"
    )
    
    cohort = st.text_input("Cohort Name", value="Cohort 2024", help="Training cohort identifier")

with col2:
    st.subheader("Demographics")
    
    gender = st.radio("Gender", ["Male", "Female"])
    
    youth = st.radio("Age Group", ["18-35 (Youth)", "35+"])
    
    grad_year = st.number_input("Graduation Year", min_value=2020, max_value=2026, value=2024)
    
    grad_month = st.slider("Graduation Month", min_value=1, max_value=12, value=6)

# Prediction button
if st.button("🔮 Predict Placement Status", key="predict_btn", use_container_width=True):
    # Prepare input (in same order as training features)
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Program Name': [program],
        'Sponsorship Type': [sponsorship],
        'Education Level': [education],
        'Graduation Year': [grad_year],
        'Graduation Month': [grad_month],
        'Cohort': [cohort]
    })
    
    # Encode categorical features
    input_encoded = input_data.copy()
    for col in input_data.columns:
        if col in encoders:
            try:
                input_encoded[col] = encoders[col].transform(input_data[col])
            except ValueError:
                st.warning(f"⚠️ Unexpected value in {col}: {input_data[col].values[0]}")
                st.stop()
    
    # Make prediction
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[:, 1][0]
    
    # Display result
    st.divider()
    st.subheader("📊 Prediction Result")
    
    if prediction[0] == 1:
        st.success(
            f"""
            ### ✅ Likely to Be Employed
            **Confidence:** {probability:.1%}
            
            This graduate shows characteristics associated with successful employment.
            """
        )
    else:
        st.warning(
            f"""
            ### ⚠️ May Need Career Support
            **Employment probability:** {probability:.1%}
            
            This graduate may benefit from:
            - Targeted career counseling
            - Mock interview practice
            - Employer networking opportunities
            """
        )
    
    # Explanation
    st.markdown("""
    ---
    **Model Note:** This prediction is based on training program, education level, 
    sponsorship type, and graduation timing. It's a decision support tool, not a 
    definitive outcome — use it alongside human judgment.
    """)

# Footer
st.divider()
st.markdown("""
**Data:** Refactory Academy Graduate Placement Dataset (2020–2024)  
**Model:** Random Forest Classifier (77% accuracy)  
**Built with:** Python, Pandas, Scikit-learn, Streamlit
""")
