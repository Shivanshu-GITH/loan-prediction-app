"""
Loan Approval Prediction - Deployment App (Streamlit)
Author : Shivanshu Tiwari (Roll No: 2400270130173)

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# ----------------------------- UI -----------------------------
st.title("🏦 Loan Approval Prediction")
st.caption("Decision Tree Classifier | Shivanshu Tiwari — 2400270130173")

st.markdown(
    "Enter the applicant details below and click **Predict** to check "
    "whether the loan is likely to be **Approved** or **Rejected**."
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=25, step=1)
    income = st.number_input("Annual Income (₹)", min_value=0, max_value=2_000_000, value=60_000, step=1_000)
with col2:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, max_value=1_000_000, value=20_000, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)

if st.button("Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[age, income, loan_amount, credit_score]],
        columns=["Age", "Income", "LoanAmount", "CreditScore"],
    )
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Loan APPROVED  (confidence: {proba[1] * 100:.1f}%)")
    else:
        st.error(f"❌ Loan REJECTED  (confidence: {proba[0] * 100:.1f}%)")

    st.progress(float(proba[1]), text=f"Approval probability: {proba[1] * 100:.1f}%")

    with st.expander("View input summary"):
        st.dataframe(input_df, hide_index=True, use_container_width=True)

# ----------------------- Model info sidebar -----------------------
with st.sidebar:
    st.header("About the Model")
    st.write(
        """
        - **Algorithm:** Decision Tree Classifier
        - **Features:** Age, Income, Loan Amount, Credit Score
        - **Training data:** 45,000 loan records
        - **Preprocessing:** duplicate removal, stratified split, standard scaling
        - **Tuning:** GridSearchCV (5-fold CV)
        """
    )
    st.divider()
    st.caption("AI & DS Internship and Training Program")
