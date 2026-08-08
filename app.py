import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="Graduate Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Load model and supporting files
# =====================================================================
with open("graduate_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

try:
    historical_data = pd.read_pickle("historical_placement_data.pkl")
except FileNotFoundError:
    historical_data = None

# =====================================================================
# Styling — clean, high-contrast, no background image
# =====================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        background-color: white;
        border-radius: 10px;
    }
    .header-container {
        background: linear-gradient(135deg, #0F4C81 0%, #1B5E9E 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .header-container h1 {
        margin: 0;
        font-size: 2.2em;
    }
    .prediction-success {
        background-color: #D4EDDA;
        border: 3px solid #28A745;
        padding: 20px;
        border-radius: 8px;
        color: #155724;
    }
    .prediction-risk {
        background-color: #F8D7DA;
        border: 3px solid #DC3545;
        padding: 20px;
        border-radius: 8px;
        color: #721C24;
    }
    .support-box {
        background-color: #E7F3FF;
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Header
# =====================================================================
st.markdown("""
<div class="header-container">
    <h1>🎓 Graduate Placement Prediction</h1>
    <p>Internal tool for Refactory Placement Officers — decision support, not a final decision.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the graduate's information below to predict the likelihood of employment "
    "and get recommended support actions."
)

# =====================================================================
# Sidebar — context
# =====================================================================
with st.sidebar:
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    **Predictions based on:**
    - Program
    - Sponsorship type
    - Education level
    - Youth status (18-35)
    - Gender
    - Graduation timing

    **NOT based on (fairness-by-design):**
    - Nationality
    - Refugee status
    - Disability status

    These are captured separately, only to route graduates to
    specialized support — never fed into the prediction.
    """)

    if historical_data is not None:
        st.markdown("### 📊 Quick Stats")
        overall_rate = (historical_data["Placement Status"] == "Employed").mean()
        st.metric("Overall placement rate", f"{overall_rate:.0%}")
        st.metric("Total graduates on file", len(historical_data))

    st.markdown("### ♿ Accessibility")
    st.write("""
    ✓ Screen reader compatible
    ✓ High contrast, no color-only signals
    ✓ Keyboard navigable
    """)

# =====================================================================
# Tabs
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
        gender = st.selectbox("Gender", list(encoders["Gender"].classes_))
        youth = st.selectbox("Youth (18-35)", list(encoders["Youth (18-35)"].classes_))

    with col2:
        st.markdown("#### 🎓 Program & Sponsorship")
        program = st.selectbox("Program Name", list(encoders["Program Name"].classes_))
        sponsorship = st.selectbox("Sponsorship Type", list(encoders["Sponsorship Type"].classes_))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 📚 Education & Cohort")
        cohort = st.selectbox("Cohort", list(encoders["Cohort"].classes_))
        education = st.selectbox("Education Level", list(encoders["Education Level"].classes_))

    with col4:
        st.markdown("#### 📅 Graduation Details")
        grad_year = st.number_input("Graduation Year", min_value=2018, max_value=2035, value=2025)
        grad_month = st.slider("Graduation Month", 1, 12, 6)

    st.markdown("---")
    st.markdown("#### 🌍 Support Routing (Optional)")
    st.caption("Used only to route graduates to specialized support — NOT fed into the prediction model")

    col5, col6 = st.columns(2)
    with col5:
        refugee = st.selectbox("Refugee Status", list(encoders["Refugee Status"].classes_))
    with col6:
        disability = st.selectbox("Disability Status", list(encoders["Disability status"].classes_))

    st.markdown("---")
    predict_btn = st.button("🔍 Generate Prediction", use_container_width=True, type="primary")

    if predict_btn:
        # Only features the model actually trained on go into the prediction.
        # Refugee/Disability/Nationality are intentionally excluded here.
        input_dict = {
            "Gender": encoders["Gender"].transform([gender])[0],
            "Youth (18-35)": encoders["Youth (18-35)"].transform([youth])[0],
            "Program Name": encoders["Program Name"].transform([program])[0],
            "Sponsorship Type": encoders["Sponsorship Type"].transform([sponsorship])[0],
            "Cohort": encoders["Cohort"].transform([cohort])[0],
            "Education Level": encoders["Education Level"].transform([education])[0],
            "Graduation Year": grad_year,
            "Graduation Month": grad_month,
        }

        input_data = pd.DataFrame([input_dict])

        try:
            input_data = input_data[feature_columns]
        except KeyError:
            st.error(f"Column mismatch. Model expects: {feature_columns}")
            st.error(f"Available: {list(input_data.columns)}")
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
            </div>
            """, unsafe_allow_html=True)
            st.success("No immediate action required — continue standard placement support.")
        else:
            st.markdown(f"""
            <div class="prediction-risk">
                <h3>⚠️ At Risk</h3>
                <p>Confidence: <strong>{1-proba:.0%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 💡 Recommended Support Actions")
            recs = []
            if education in ["Unknown", "Secondary (O-Level)", "Secondary (A-Level)"]:
                recs.append("📚 Offer skills certification or upskilling pathway.")
            if sponsorship in ["10X Program", "Self-sponsored"]:
                recs.append(f"🤝 Prioritize employer-matching outreach ('{sponsorship}' historically needs more support).")
            if youth == "No":
                recs.append("📍 Check whether non-youth-targeted employer programs apply.")
            if not recs:
                recs.append("📞 Schedule a check-in call to identify individual barriers (job search skills, interview readiness, network gaps).")
            for r in recs:
                st.write(f"- {r}")

        if refugee == "Yes" or disability == "Yes":
            st.markdown("""
            <div class="support-box">
                <h3>🌍 Specialized Support Routing</h3>
                <p>This graduate has disclosed refugee and/or disability status.</p>
                <p><strong>Action:</strong> Route to Refactory's specialized placement support
                track for tailored employer matching and any needed accommodations.</p>
                <p><strong>Important:</strong> Do not assume the cause of non-placement is
                related to this status — treat this as a support-routing flag, not a diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("This prediction is decision support for Refactory staff, not a final determination.")

# =====================================================================
# TAB 2: ANALYTICS
# =====================================================================
with tab2:
    st.markdown("### 📊 Historical Placement Analytics")

    if historical_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Placement Rate by Program")
            rate = historical_data.groupby("Program Name")["Placement Status"].apply(
                lambda x: (x == "Employed").mean()
            ).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            rate.plot(kind="barh", ax=ax, color="#0F4C81")
            ax.set_xlabel("Employment Rate")
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("#### Placement Rate by Sponsorship")
            rate2 = historical_data.groupby("Sponsorship Type")["Placement Status"].apply(
                lambda x: (x == "Employed").mean()
            ).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            rate2.plot(kind="barh", ax=ax, color="#28A745")
            ax.set_xlabel("Employment Rate")
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning(
            "Historical data not found. In the notebook, run:\n\n"
            "`historical_data = df.copy()`\n\n"
            "`historical_data.to_pickle('historical_placement_data.pkl')`\n\n"
            "and upload that file alongside this app."
        )

# =====================================================================
# TAB 3: ABOUT
# =====================================================================
with tab3:
    st.markdown("### How This Model Works")
    st.write("""
    **Model:** Random Forest Classifier
    **Accuracy:** 77% on held-out test data

    **Features used:** Program, Sponsorship Type, Education Level, Youth status,
    Gender, Graduation Year/Month, Cohort

    **Deliberately NOT used:** Nationality, Refugee Status, Disability Status
    (fairness-by-design — captured separately for support routing only)
    """)

    st.markdown("### ⚠️ Limitations")
    st.warning("""
    - Reflects historical patterns; future labor market may differ
    - Individual factors (motivation, interview skills, networks) aren't in the data
    - Decision-support only — always pair with human judgement
    """)
