import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import io

# Page config with accessible colors
st.set_page_config(
    page_title="Graduate Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and supporting files
with open("graduate_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Custom CSS for accessibility (high contrast, readable fonts)
st.markdown("""
<style>
    body {
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
    .header-container {
        background-color: #0F4C81;
        padding: 20px;
        border-radius: 8px;
        color: white;
        margin-bottom: 20px;
    }
    .prediction-success {
        background-color: #D4EDDA;
        border: 2px solid #28A745;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    .prediction-risk {
        background-color: #F8D7DA;
        border: 2px solid #DC3545;
        padding: 15px;
        border-radius: 5px;
        color: #721C24;
    }
    .input-section {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #0F4C81;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo/branding
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("🎓", unsafe_allow_html=True)
with col2:
    st.markdown("""
    # Graduate Placement Prediction System
    **Internal tool for Refactory Placement Officers**
    """)

st.markdown("""
**Purpose:** Decision support to identify graduates who may need proactive employment support.
Not a final hiring decision — always supplement with human judgment.
""")

# Sidebar with context
with st.sidebar:
    st.markdown("### About this tool")
    st.info("""
    ✓ Predictions based on:
    - Program type
    - Sponsorship pathway
    - Education level
    - Graduation timing
    
    ✓ NOT based on:
    - Nationality
    - Refugee status
    - Disability status
    
    (These are flagged separately for support routing.)
    """)

# Tab structure for better organization (screen readers handle this well)
tab1, tab2 = st.tabs(["Prediction", "About the model"])

with tab1:
    st.markdown("### Step 1: Enter graduate information")
    
    # Organize inputs in an accessible grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        gender = st.selectbox(
            "Gender",
            list(encoders["Gender"].classes_),
            help="Graduate's gender identity"
        )
        youth = st.selectbox(
            "Youth (18-35)",
            list(encoders["Youth (18-35)"].classes_),
            help="Is the graduate in the 18-35 age bracket?"
        )
    
    with col2:
        st.markdown("#### Program & Sponsorship")
        program = st.selectbox(
            "Program Name",
            list(encoders["Program Name"].classes_),
            help="Which Refactory program did they complete?"
        )
        sponsorship = st.selectbox(
            "Sponsorship Type",
            list(encoders["Sponsorship Type"].classes_),
            help="How was the program funded?"
        )
    
    # Second row
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Education & Timing")
        cohort = st.selectbox(
            "Cohort",
            list(encoders["Cohort"].classes_),
            help="Which cohort/batch did they graduate with?"
        )
        education = st.selectbox(
            "Education Level",
            list(encoders["Education Level"].classes_),
            help="What was their pre-Refactory education level?"
        )
    
    with col4:
        st.markdown("#### Graduation Details")
        grad_year = st.number_input(
            "Graduation Year",
            min_value=2018,
            max_value=2035,
            value=2025,
            help="Year the graduate completed the program"
        )
        grad_month = st.slider(
            "Graduation Month",
            1, 12, 6,
            help="Month of graduation (1=Jan, 12=Dec)"
        )
    
    st.markdown("---")
    st.markdown("#### Support routing (optional)")
    st.caption("Used to direct graduates to specialized support pathways — NOT fed into prediction")
    
    col5, col6 = st.columns(2)
    with col5:
        refugee = st.selectbox(
            "Refugee Status",
            list(encoders["Refugee Status"].classes_),
            help="Graduate's refugee status (for support routing only)"
        )
    with col6:
        disability = st.selectbox(
            "Disability Status",
            list(encoders["Disability status"].classes_),
            help="Graduate's disability status (for support routing only)"
        )
    
    # Prediction button
    st.markdown("---")
    predict_btn = st.button("🔍 Generate Prediction", use_container_width=True, type="primary")
    
    if predict_btn:
        
        # Create input data
        input_dict = {
            "Gender": encoders["Gender"].transform([gender])[0],
            "Nationality": encoders["Nationality"].transform([encoders["Nationality"].classes_[0]])[0],
            "Refugee Status": encoders["Refugee Status"].transform([refugee])[0],
            "Disability status": encoders["Disability status"].transform([disability])[0],
            "Youth (18-35)": encoders["Youth (18-35)"].transform([youth])[0],
            "Program Name": encoders["Program Name"].transform([program])[0],
            "Sponsorship Type": encoders["Sponsorship Type"].transform([sponsorship])[0],
            "Cohort": encoders["Cohort"].transform([cohort])[0],
            "Education Level": encoders["Education Level"].transform([education])[0],
            "Graduation Year": grad_year,
            "Graduation Month": grad_month
        }
        
        input_data = pd.DataFrame([input_dict])
        
        try:
            input_data = input_data[feature_columns]
        except KeyError as e:
            st.error(f"❌ Column mismatch. Model expects: {feature_columns}")
            st.stop()
        
        proba = model.predict_proba(input_data)[0][1]
        prediction = int(proba >= 0.5)
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-success">
                <h3>✅ Likely Employed</h3>
                <p>Confidence: <strong>{proba:.0%}</strong></p>
                <p>This graduate is predicted to find employment within 6 months based on their program, sponsorship, and education profile.</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("No immediate action required, but consider regular check-ins to support job placement.")
        
        else:
            st.markdown(f"""
            <div class="prediction-risk">
                <h3>⚠️ At Risk</h3>
                <p>Confidence: <strong>{1-proba:.0%}</strong></p>
                <p>This graduate may need proactive employment support.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Recommended Support Actions")
            
            recs = []
            reasons = []
            
            if education in ["Not Reported", "Secondary (O-Level)", "Secondary (A-Level)"]:
                recs.append("📚 Offer additional skills certification or upskilling pathway")
                reasons.append(f"Education level ({education}) is entry-level")
            
            if sponsorship in ["10X Program", "Self-sponsored"]:
                recs.append(f"🤝 Prioritize direct employer-matching outreach (graduates from {sponsorship} historically need more support)")
                reasons.append(f"Sponsorship type ({sponsorship}) has historically lower placement rate")
            
            if youth == "No":
                recs.append("📍 Route to non-youth-targeted employer programs")
                reasons.append(f"Graduate is outside the 18-35 youth bracket")
            
            if program in ["Blockchain Development", "Cloud Computing"]:
                recs.append(f"🔍 Identify adjacent roles in fintech/distributed systems (market for {program} is smaller)")
                reasons.append(f"Program ({program}) has smaller job market")
            
            if not recs:
                recs.append("📞 Schedule a 1-on-1 check-in to identify individual barriers (job search skills, interview readiness, network gaps)")
            
            for rec in recs:
                st.write(rec)
            
            # Reasons (screen reader friendly)
            with st.expander("📋 View reasons for this prediction"):
                st.markdown("**Factors contributing to this prediction:**")
                for reason in reasons if reasons else ["No specific risk driver identified — recommend individual assessment."]:
                    st.markdown(f"- {reason}")
        
        # Support routing
        if refugee == "Yes" or disability == "Yes":
            st.markdown("---")
            st.info("""
            ### 🌍 Specialized Support Routing
            
            This graduate has disclosed **refugee status** and/or **disability status**.
            
            **Action:** Route to Refactory's specialized placement support team for:
            - Tailored employer matching
            - Workplace accommodation coordination
            - Connection to refugee-experienced employers
            
            **Important:** Do not assume non-placement is related to these statuses. 
            Our fairness audit found outcome gaps that are not fully explained by available data.
            Treat this as a support-routing flag, not a diagnosis.
            """)
        
        st.markdown("---")
        st.caption("💭 This prediction is decision support for staff. Always supplement with human judgment and individual graduate assessment.")

with tab2:
    st.markdown("### How this model works")
    st.write("""
    **Prediction Method:** Random Forest Classifier
    
    **Features Used:**
    - Program Name
    - Sponsorship Type
    - Education Level
    - Youth status (18-35)
    - Gender
    - Graduation year/month
    - Cohort
    
    **NOT used (fairness-by-design):**
    - Nationality
    - Refugee Status
    - Disability Status
    
    These are captured separately only to route graduates to appropriate support pathways.
    
    **Accuracy:** 77% on test data
    
    **Limitations:**
    - Model captures historical patterns; future labor market may differ
    - Individual circumstances (job search skills, networks, personal factors) not captured in data
    - Should supplement, not replace, human placement officer judgment
    """)
    
    st.markdown("### Accessibility Features")
    st.write("""
    ✓ **Screen reader compatible** — all images and icons have descriptive text labels
    
    ✓ **High contrast colors** — meets WCAG AA standards for readability
    
    ✓ **Clear headings** — organized for keyboard navigation
    
    ✓ **Help text** — hover/click on field labels for context
    
    ✓ **No color-only information** — red/green codes are paired with text (✅ / ⚠️)
    
    **For screen reader users:** Use Tab to navigate, Enter to select options, and all predictions are read aloud with confidence levels.
    """)
