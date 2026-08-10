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
        refugee = st.selectbox("Refugee Status", ["No", "Yes"])
    with col6:
        disability = st.selectbox("Disability Status", ["No", "Yes"])

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

            st.markdown("### 💡 Recommended Next Steps")
            if proba >= 0.75:
                st.success(
                    "**Strong signal — light-touch support.** This profile closely matches "
                    "graduates who placed successfully. Standard next steps:\n"
                    "- Add to the alumni job board / employer referral list\n"
                    "- One check-in at the 30-day mark to confirm placement landed\n"
                    "- No intensive intervention needed — focus officer time on lower-confidence cases"
                )
            else:
                st.info(
                    "**Positive but not certain — light monitoring recommended.** "
                    f"Confidence is {proba:.0%}, above the risk threshold but not by a wide margin. "
                    "Standard next steps:\n"
                    "- Include in the next cohort's group job-readiness session (interview prep, CV review)\n"
                    "- Connect with 1-2 alumni mentors in their program area for informal networking\n"
                    "- Re-check status at 60 days — if still unplaced, treat as at-risk and escalate to "
                    "individual outreach"
                )
        else:
            st.markdown(f"""
            <div class="prediction-risk">
                <h3>⚠️ At Risk</h3>
                <p>Confidence: <strong>{1-proba:.0%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 💡 Recommended Support Plan")

            plans = []

            if education in ["Unknown", "Secondary (O-Level)", "Secondary (A-Level)"]:
                plans.append({
                    "title": "📚 Credential Gap — Upskilling Pathway",
                    "why": f"Education level on file ({education}) is entry-level, which can filter this graduate out of employer screening before skills are even assessed.",
                    "steps": [
                        "Book a skills-audit session within 5 business days to identify a certificate or micro-credential that closes the gap fastest",
                        "Compile a portfolio of Refactory project work — something concrete to show alongside the credential",
                        "Add graduate to the next available portfolio-review workshop",
                        "Target: enrolled in an upskilling track within 2 weeks",
                    ]
                })

            if sponsorship in ["10X Program", "Self-sponsored"]:
                plans.append({
                    "title": f"🤝 Employer Access — '{sponsorship}' Pathway",
                    "why": f"Graduates on '{sponsorship}' sponsorship historically have lower placement rates, likely due to fewer built-in employer partnerships compared to other sponsorship tracks.",
                    "steps": [
                        "Schedule a 1:1 with a placement officer within 1 week (not the general job board)",
                        "Identify 3-5 target employers based on program + location fit",
                        "Prepare one tailored application per employer, not a generic CV blast",
                        "Officer follows up with each employer contact at the 2-week mark",
                    ]
                })

            if youth == "No":
                plans.append({
                    "title": "📍 Program Eligibility Review",
                    "why": "Graduate is outside the 18-35 youth bracket, so youth-targeted employer partnerships and funding programs may not apply.",
                    "steps": [
                        "Cross-check this graduate against Refactory's non-youth employer partner list",
                        "If no non-youth pathway exists yet for their program, flag to program lead as a gap",
                        "Route to general (non-age-restricted) placement channels in the meantime",
                    ]
                })

            if not plans:
                plans.append({
                    "title": "📞 Individual Assessment",
                    "why": "No structural risk factor stood out — education, sponsorship, and age bracket all look typical for a successfully-placed graduate. The barrier is likely individual.",
                    "steps": [
                        "Call within 1 week — open-ended: 'what's been your experience applying so far?'",
                        "Based on their answer, route to: interview prep, CV/portfolio review, or a recruiter/alum network introduction",
                        "Re-check status at 30 days",
                    ]
                })

            for plan in plans:
                with st.container():
                    st.markdown(f"**{plan['title']}**")
                    st.caption(plan["why"])
                    for i, step in enumerate(plan["steps"], 1):
                        st.markdown(f"{i}. {step}")
                    st.markdown("")

        if refugee == "Yes" or disability == "Yes":
            st.markdown("---")
            st.markdown("### 🌍 Specialized Support Checklist")
            st.caption(
                "These are procedural steps within Refactory's control — not assumptions "
                "about why this graduate hasn't been placed. Our fairness audit found no "
                "reliable link between these statuses and placement outcomes, so treat this "
                "as a standard support checklist, not a diagnosis."
            )

            if refugee == "Yes":
                with st.container():
                    st.markdown("**🪪 Refugee Status — Documentation & Access Checklist**")
                    st.markdown("""
1. Confirm work permit / right-to-work documentation is current and on file — if expired or pending, flag to legal/admin support immediately, since this is often the actual blocker, not skill
2. Verify prior credentials (secondary/university) have been through any required local recognition or equivalency process — assist with this if not yet done
3. Match to Refactory's employer partners with **prior experience hiring refugee talent**, so this isn't the employer's first case
4. If no such employer exists for this graduate's program, loop in the partnerships team to identify one — this is a gap to close, not work around
5. Provide reference/verification support directly (Refactory vouching) if the graduate's local reference network is limited
                    """)

            if disability == "Yes":
                with st.container():
                    st.markdown("**♿ Disability Status — Accessibility & Accommodation Checklist**")
                    st.markdown("""
1. Ask the graduate directly what accommodations (if any) they'd want during interviews and on the job — do not assume; this varies per person
2. Offer accessible interview logistics as standard: remote/flexible-time option, materials in accessible format
3. Check employer eligibility for Uganda's Persons with Disabilities Act workplace accommodation support/subsidies, and raise this with the employer upfront so accommodation cost isn't framed as solely theirs to bear
4. Add employer to (or check against) Refactory's list of employers with positive prior experience hiring graduates with disabilities
5. Follow up at 2 weeks post-introduction specifically to check whether any accommodation gap emerged that needs addressing
                    """)

        st.caption("This prediction is decision support for Refactory staff, not a final determination.")

# =====================================================================
# TAB 2: ANALYTICS DASHBOARD
# =====================================================================
with tab2:
    st.markdown("### 📊 Placement Analytics Dashboard")

    if historical_data is not None:
        hd = historical_data.copy()
        hd["is_employed"] = (hd["Placement Status"] == "Employed").astype(int)

        # -------------------------------------------------------------
        # ROW 1 — KPI cards
        # -------------------------------------------------------------
        overall_rate = hd["is_employed"].mean()
        total_grads = len(hd)
        at_risk_count = int((hd["is_employed"] == 0).sum())

        # Best / worst program by placement rate (min sample size 15)
        prog_counts = hd["Program Name"].value_counts()
        valid_progs = prog_counts[prog_counts >= 15].index
        prog_rate = hd[hd["Program Name"].isin(valid_progs)].groupby("Program Name")["is_employed"].mean()
        best_prog = prog_rate.idxmax()
        worst_prog = prog_rate.idxmin()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Placement Rate", f"{overall_rate:.0%}")
        k2.metric("Total Graduates", f"{total_grads:,}")
        k3.metric("Currently At-Risk", f"{at_risk_count:,}", delta=f"{at_risk_count/total_grads:.0%} of total", delta_color="inverse")
        k4.metric("Best Program", best_prog, delta=f"{prog_rate[best_prog]:.0%} placed")

        st.markdown("---")

        # -------------------------------------------------------------
        # ROW 2 — Placement rate by Program & Sponsorship
        # -------------------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Placement Rate by Program")
            rate = hd[hd["Program Name"].isin(valid_progs)].groupby("Program Name")["is_employed"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ["#DC3545" if v < 0.5 else "#FFC107" if v < 0.65 else "#28A745" for v in rate.values]
            rate.plot(kind="barh", ax=ax, color=colors)
            ax.set_xlabel("Employment Rate")
            ax.set_xlim(0, 1)
            for i, v in enumerate(rate.values):
                ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("#### Placement Rate by Sponsorship")
            spon_counts = hd["Sponsorship Type"].value_counts()
            valid_spon = spon_counts[spon_counts >= 15].index
            rate2 = hd[hd["Sponsorship Type"].isin(valid_spon)].groupby("Sponsorship Type")["is_employed"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 5))
            colors2 = ["#DC3545" if v < 0.5 else "#FFC107" if v < 0.65 else "#28A745" for v in rate2.values]
            rate2.plot(kind="barh", ax=ax, color=colors2)
            ax.set_xlabel("Employment Rate")
            ax.set_xlim(0, 1)
            for i, v in enumerate(rate2.values):
                ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")

        # -------------------------------------------------------------
        # ROW 3 — Trend over time + Education level
        # -------------------------------------------------------------
        col3, col4 = st.columns(2)

        with col3:
            if "Graduation Year" in hd.columns:
                st.markdown("#### Placement Rate Trend by Cohort Year")
                trend = hd.groupby("Graduation Year")["is_employed"].mean().sort_index()
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(trend.index.astype(str), trend.values, marker="o", linewidth=2, color="#0F4C81")
                ax.set_ylabel("Employment Rate")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.3)
                for x, v in zip(trend.index.astype(str), trend.values):
                    ax.annotate(f"{v:.0%}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

        with col4:
            st.markdown("#### Placement Rate by Education Level")
            edu_counts = hd["Education Level"].value_counts()
            valid_edu = edu_counts[edu_counts >= 15].index
            rate3 = hd[hd["Education Level"].isin(valid_edu)].groupby("Education Level")["is_employed"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors3 = ["#28A745" if v >= 0.6 else "#FFC107" if v >= 0.5 else "#DC3545" for v in rate3.values]
            rate3.plot(kind="bar", ax=ax, color=colors3)
            ax.set_ylabel("Employment Rate")
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")

        # -------------------------------------------------------------
        # ROW 4 — Actionable business decision box
        # -------------------------------------------------------------
        st.markdown("### 🎯 What This Means for Placement Strategy")
        st.markdown(f"""
        <div class="support-box">
            <p><strong>{at_risk_count} graduates ({at_risk_count/total_grads:.0%})</strong> are currently
            flagged as not-yet-employed.</p>
            <p><strong>Priority action:</strong> <em>{worst_prog}</em> has the lowest placement rate
            ({prog_rate[worst_prog]:.0%}) among programs with sufficient data — prioritize employer
            outreach and curriculum-market alignment review for this program first.</p>
            <p><strong>What's working:</strong> <em>{best_prog}</em> leads placement at
            {prog_rate[best_prog]:.0%} — study its employer partnerships as a template for
            lower-performing programs.</p>
        </div>
        """, unsafe_allow_html=True)

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

    st.markdown("### ♿ Accessibility")
    st.info("""
    - **Screen reader compatible** — all inputs have descriptive labels, headings use semantic structure
    - **High contrast, no color-only signals** — status is always paired with text/icons (✅ / ⚠️), not color alone
    - **Keyboard navigable** — Tab through inputs, Enter to select, works without a mouse
    """)
