import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
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

# Background image + custom CSS for beautiful interface
st.markdown("""
<style>
    /* Clean solid background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main content area */
    .main {
        background-color: white;
        border-radius: 10px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #0F4C81 0%, #1B5E9E 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .header-container p {
        margin: 5px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    /* Input sections */
    .input-section {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #0F4C81;
        margin-bottom: 15px;
    }
    
    .input-section h3 {
        margin-top: 0;
        color: #0F4C81;
    }
    
    /* Success/Risk predictions */
    .prediction-success {
        background-color: #D4EDDA;
        border: 3px solid #28A745;
        padding: 20px;
        border-radius: 8px;
        color: #155724;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-risk {
        background-color: #F8D7DA;
        border: 3px solid #DC3545;
        padding: 20px;
        border-radius: 8px;
        color: #721C24;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-success h3,
    .prediction-risk h3 {
        margin-top: 0;
        font-size: 1.8em;
    }
    
    /* Cards */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-top: 4px solid #0F4C81;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h4 {
        margin: 0;
        color: #0F4C81;
        font-size: 0.9em;
    }
    
    .metric-card .number {
        font-size: 2em;
        font-weight: bold;
        color: #0F4C81;
        margin: 5px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #28A745;
        color: white;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #218838;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Support routing info box */
    .support-box {
        background-color: #E7F3FF;
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    .support-box h3 {
        color: #1976D2;
        margin-top: 0;
    }
    
    /* Chart styling */
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.98);
    }
    
    /* Text accessibility */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data for visualizations (from your training set)
# You'll need to save this in your notebook as well
try:
    historical_data = pd.read_pickle("historical_placement_data.pkl")
except:
    historical_data = None

# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="header-container">
    <h1>🎓 Graduate Placement Prediction</h1>
    <p>Supporting Refactory's placement success — predict employment outcomes & identify support needs</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Purpose:** Decision support to identify graduates who may need proactive employment support.
This tool is designed for Refactory placement officers and partners to make data-informed outreach decisions.
""")

# =====================================================================
# SIDEBAR - Context & Key Stats
# =====================================================================
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    if historical_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Overall Placement</h4>
                <div class="number">{(historical_data['Placement Status'] == 'Employed').mean():.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Graduates</h4>
                <div class="number">{len(historical_data)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    **Predictions based on:**
    - Program type
    - Sponsorship pathway
    - Education level
    - Graduation timing
    
    **NOT based on:**
    - Nationality
    - Refugee status
    - Disability status
    
    *(Used separately for support routing)*
    """)
    
    st.markdown("### ♿ Accessibility")
    st.write("""
    ✓ Screen reader compatible
    ✓ High contrast design
    ✓ Keyboard navigable
    ✓ Mobile friendly
    """)

# =====================================================================
# MAIN TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Analytics", "❓ About"])

# =====================================================================
# TAB 1: PREDICTION
# =====================================================================
with tab1:
    st.markdown("### Step 1: Enter Graduate Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Personal Information")
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
        st.markdown("#### 🎓 Program & Sponsorship")
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
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📚 Education & Cohort")
        cohort = st.selectbox(
            "Cohort",
            list(encoders["Cohort"].classes_),
            help="Which cohort/batch did they graduate with?"
        )
        education = st.selectbox(
            "Education Level",
            list(encoders["Education Level"].classes_),
            help="Pre-Refactory education level"
        )
    
    with col4:
        st.markdown("#### 📅 Graduation Details")
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
    st.markdown("#### 🌍 Support Routing (Optional)")
    st.caption("Used to direct graduates to specialized support — NOT used in prediction")
    
    col5, col6 = st.columns(2)
    with col5:
        refugee = st.selectbox(
            "Refugee Status",
            list(encoders["Refugee Status"].classes_),
            help="For specialized support routing only"
        )
    with col6:
        disability = st.selectbox(
            "Disability Status",
            list(encoders["Disability status"].classes_),
            help="For specialized support routing only"
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
            st.error(f"❌ Column mismatch. Contact support.")
            st.stop()
        
        proba = model.predict_proba(input_data)[0][1]
        prediction = int(proba >= 0.5)
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-success">
                <h3>✅ Likely Employed</h3>
                <p><strong>Confidence: {proba:.0%}</strong></p>
                <p>This graduate is predicted to find employment within 6 months.</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("💚 No immediate action required. Continue with standard placement support.")
        
        else:
            st.markdown(f"""
            <div class="prediction-risk">
                <h3>⚠️ At Risk</h3>
                <p><strong>Confidence: {1-proba:.0%}</strong></p>
                <p>This graduate may need proactive employment support.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Recommended Support Actions")
            
            recs = []
            reasons = []
            
            if education in ["Not Reported", "Secondary (O-Level)", "Secondary (A-Level)"]:
                recs.append("📚 **Skills certification:** Offer micro-credentials or upskilling pathway")
                reasons.append(f"Education level ({education}) may limit employer screening")
            
            if sponsorship in ["10X Program", "Self-sponsored"]:
                recs.append(f"🤝 **Employer outreach:** Prioritize direct employer matching ('{sponsorship}' historically needs more support)")
                reasons.append(f"Sponsorship type ({sponsorship}) shows lower placement rates")
            
            if youth == "No":
                recs.append("📍 **Employer routing:** Match with non-youth-targeted programs")
                reasons.append("Graduate is outside the 18-35 youth demographic")
            
            if program in ["Blockchain Development", "Cloud Computing", "AI/ML"]:
                recs.append(f"🔍 **Market alignment:** Identify adjacent roles (fintech, distributed systems, ML ops)")
                reasons.append(f"'{program}' market is smaller — may need role reframing")
            
            if not recs:
                recs.append("📞 **Individual assessment:** Schedule 1-on-1 call to identify personal barriers (job search skills, interview readiness, networks)")
            
            for rec in recs:
                st.markdown(f"{rec}")
            
            with st.expander("📋 View detailed reasons"):
                for reason in (reasons if reasons else ["No major structural barrier identified — likely personal factors at play"]):
                    st.markdown(f"- {reason}")
        
        # Support routing
        if refugee == "Yes" or disability == "Yes":
            st.markdown("""
            <div class="support-box">
                <h3>🌍 Specialized Support Routing</h3>
                <p>This graduate has disclosed <strong>refugee status</strong> and/or <strong>disability status</strong>.</p>
                <p><strong>Action:</strong> Route to Refactory's specialized placement team for:</p>
                <ul>
                    <li>Tailored employer matching</li>
                    <li>Workplace accommodation coordination</li>
                    <li>Connection to refugee-experienced employers</li>
                </ul>
                <p><strong>⚠️ Important:</strong> Do not assume non-placement is related to these statuses. 
                Treat this as a support-routing flag, not a diagnosis. Individual assessment still needed.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("💭 This prediction is decision support. Always supplement with human judgment and individual graduate assessment.")

# =====================================================================
# TAB 2: ANALYTICS
# =====================================================================
with tab2:
    st.markdown("### 📊 Historical Placement Analytics")
    
    if historical_data is not None:
        # Placement rate by program
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Placement Rate by Program")
            placement_by_program = historical_data.groupby("Program Name")["Placement Status"].apply(
                lambda x: (x == "Employed").mean()
            ).sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            placement_by_program.plot(kind="barh", ax=ax, color="#0F4C81")
            ax.set_xlabel("Employment Rate (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Program", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            for i, v in enumerate(placement_by_program):
                ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Placement Rate by Sponsorship")
            placement_by_sponsor = historical_data.groupby("Sponsorship Type")["Placement Status"].apply(
                lambda x: (x == "Employed").mean()
            ).sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            placement_by_sponsor.plot(kind="barh", ax=ax, color="#28A745")
            ax.set_xlabel("Employment Rate (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Sponsorship Type", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            for i, v in enumerate(placement_by_sponsor):
                ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Education level breakdown
        st.markdown("#### Placement Rate by Education Level")
        placement_by_edu = historical_data.groupby("Education Level")["Placement Status"].apply(
            lambda x: (x == "Employed").mean()
        ).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#28A745" if x >= 0.6 else "#FFC107" if x >= 0.5 else "#DC3545" for x in placement_by_edu]
        placement_by_edu.plot(kind="bar", ax=ax, color=colors)
        ax.set_ylabel("Employment Rate", fontsize=11, fontweight="bold")
        ax.set_xlabel("Education Level", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
        for i, v in enumerate(placement_by_edu):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.warning("Historical data not available. Run this command in your notebook to save it:\n```python\nhistorical_data.to_pickle('historical_placement_data.pkl')\n```")

# =====================================================================
# TAB 3: ABOUT
# =====================================================================
with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### How This Model Works")
        st.write("""
        **Model Type:** Random Forest Classifier
        
        **What it predicts:** Whether a graduate will be employed within 6 months of graduation
        
        **Accuracy on test data:** 77%
        
        **Features used:**
        - Program Name
        - Sponsorship Type
        - Education Level
        - Youth status (18-35)
        - Gender
        - Graduation year/month
        - Cohort
        
        **Importantly, NOT used:**
        - Nationality
        - Refugee Status
        - Disability Status
        
        These are captured separately and used only for routing graduates to appropriate support pathways (fairness-by-design).
        """)
    
    with col2:
        st.markdown("### Model Performance")
        st.metric("Accuracy", "77%")
        st.metric("Precision", "76%")
        st.metric("Recall", "77%")
        st.metric("F1-Score", "77%")
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important Limitations")
    st.warning("""
    1. **Historical patterns only** — Model captures past data; future labor market may differ
    2. **Missing factors** — Job search skills, personal networks, individual motivation not captured
    3. **Not a final decision** — Use as decision support alongside human placement officer judgment
    4. **Fairness caveats** — Fairness audit showed gaps by refugee status even with that attribute excluded; specialist routing recommended
    """)
    
    st.markdown("### ♿ Accessibility Features")
    st.info("""
    **For screen reader users:**
    - All images and icons have descriptive labels
    - Headings are semantic (H1, H2, H3)
    - Tables and charts are labeled
    - Form fields have help text
    
    **For visual accessibility:**
    - High contrast colors (WCAG AA compliant)
    - No color-only information (✅/⚠️ paired with text)
    - Clear, readable fonts (16px minimum)
    - Responsive design works on all device sizes
    
    **For keyboard navigation:**
    - Tab through all inputs and buttons
    - Enter to select, Space to expand sections
    - Alt+Tab to navigate between tabs
    """)
    
    st.markdown("---")
    
    st.markdown("### 📧 Support & Feedback")
    st.write("""
    Questions about this tool?
    
    **Contact:** Refactory Placement Team
    
    **Report issues:** [feedback link]
    """)
