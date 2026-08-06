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
st.caption("Internal tool for Refactory Placement Officers — decision support, not a final decision.")

st.write("Enter the graduate's information below to predict the likelihood of employment.")

# -----------------------------
# Prediction inputs — ONLY features the model was trained and audited on.
# Nationality / Refugee Status / Disability status are intentionally excluded
# here (fairness-by-design) and instead captured separately below, purely to
# route flagged graduates to the right support pathway — not to predict on.
# -----------------------------

gender = st.selectbox("Gender", list(encoders["Gender"].classes_))
youth = st.selectbox("Youth (18-35)", list(encoders["Youth (18-35)"].classes_))
program = st.selectbox("Program Name", list(encoders["Program Name"].classes_))
sponsorship = st.selectbox("Sponsorship Type", list(encoders["Sponsorship Type"].classes_))
cohort = st.selectbox("Cohort", list(encoders["Cohort"].classes_))
education = st.selectbox("Education Level", list(encoders["Education Level"].classes_))
grad_year = st.number_input("Graduation Year", min_value=2018, max_value=2035, value=2025)
grad_month = st.slider("Graduation Month", 1, 12, 6)

st.markdown("---")
st.caption(
    "Optional — used only to route support, never fed into the prediction model:"
)
refugee = st.selectbox("Refugee Status", list(encoders["Refugee Status"].classes_))
disability = st.selectbox("Disability Status", list(encoders["Disability status"].classes_))

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Employment"):

    input_data = pd.DataFrame([{
        "Gender": encoders["Gender"].transform([gender])[0],
        "Youth (18-35)": encoders["Youth (18-35)"].transform([youth])[0],
        "Program Name": encoders["Program Name"].transform([program])[0],
        "Sponsorship Type": encoders["Sponsorship Type"].transform([sponsorship])[0],
        "Cohort": encoders["Cohort"].transform([cohort])[0],
        "Education Level": encoders["Education Level"].transform([education])[0],
        "Graduation Year": grad_year,
        "Graduation Month": grad_month,
    }])[feature_columns]

    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= 0.5)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Predicted **likely employed** within 6 months (confidence: {proba:.0%})")
    else:
        st.error(f"❌ Predicted **at risk / not yet employed** within 6 months (confidence: {1-proba:.0%})")

        st.subheader("Recommended next steps")
        recs = []

        # Data-driven: grounded in what the model actually learned
        # (Random Forest feature_importances_), not speculation.
        if education in ["Not Reported", "Secondary (O-Level)", "Secondary (A-Level)"]:
            recs.append(
                "Education level on file is entry-level or unreported — confirm "
                "whether a certificate/diploma pathway or upskilling referral is "
                "appropriate before outreach."
            )
        if sponsorship in ["10X Program", "Self-sponsored"]:
            recs.append(
                f"Historically, graduates on '{sponsorship}' sponsorship have a "
                "lower placement rate in this dataset — prioritize this graduate "
                "for employer-matching outreach rather than passive listing."
            )
        if youth == "No":
            recs.append(
                "Graduate is outside the 18-35 youth bracket — check whether "
                "youth-targeted employer programs apply, or route to general "
                "placement channels instead."
            )
        if not recs:
            recs.append(
                "No specific risk driver stood out from program/sponsorship/education "
                "— recommend a general check-in call to identify individual barriers "
                "(e.g. job search skills, interview readiness, network gaps)."
            )

        for r in recs:
            st.write(f"- {r}")

        # Neutral support-routing flag — no causal claim about *why* they're
        # unemployed, just a route to the right specialized resource.
        if refugee == "Yes" or disability == "Yes":
            st.info(
                "📋 This graduate has disclosed refugee and/or disability status. "
                "Route to Refactory's specialized placement support track for "
                "tailored employer matching and any needed workplace "
                "accommodations coordination. Do not assume the cause of "
                "non-placement is related to this status — our fairness audit "
                "found outcome gaps here that are not fully explained by the "
                "data, so treat this as a support-routing flag, not a diagnosis."
            )

    st.caption(
        "This prediction is decision support for Refactory staff, not a final "
        "determination. See docs/fairness_audit.md before acting on flagged cases."
    )
