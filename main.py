import streamlit as st
import pandas as pd
from prediction_helper import predict

# ---------- Page Config ----------
st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

# ---------- Title ----------
st.markdown("<h1 style='text-align: center; color: #0e76a8;'>🔍 Loan Default Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Assess Probability of Default (PD) and Credit Score</h4>", unsafe_allow_html=True)
st.markdown("---")


# ---------- Section 1: Basic Inputs ----------
st.markdown("### 👤 Personal & Financial Information")
row1 = st.columns(3)

with row1[0]:
    age = st.number_input('📅 Age', min_value=18, max_value=100, step=1, value=28)

with row1[1]:
    income = st.number_input('💼 Annual Income (₹)', min_value=0, step=1000, value=1200000)

with row1[2]:
    loan_amount = st.number_input('🏦 Loan Amount (₹)', min_value=0, step=1000, value=2560000)


# ---------- Section 2: Derived Inputs ----------
loan_to_income_ratio = loan_amount / income if income > 0 else 0

st.markdown("### 💰 Loan Information")
row2 = st.columns(3)

with row2[0]:
    st.markdown("#### Loan-to-Income Ratio")
    st.markdown(f"<div style='font-size:24px; color:#336699; font-weight:bold;'>{loan_to_income_ratio:.2f}</div>", unsafe_allow_html=True)

with row2[1]:
    loan_tenure_months = st.number_input('📆 Loan Tenure (months)', min_value=0, step=1, value=36)

with row2[2]:
    avg_dpd_per_delinquency = st.number_input('📊 Avg DPD per Delinquency', min_value=0, step=1, value=20)


# ---------- Section 3: Behavior ----------
st.markdown("### 📉 Credit Behavior & Utilization")
row3 = st.columns(3)

with row3[0]:
    delinquency_ratio = st.slider('⚠️ Delinquency Ratio (%)', min_value=0, max_value=100, value=30)

with row3[1]:
    credit_utilization_ratio = st.slider('📈 Credit Utilization Ratio (%)', min_value=0, max_value=100, value=30)

with row3[2]:
    num_open_accounts = st.selectbox('📂 Open Loan Accounts', [0, 1, 2, 3, 4], index=1)


# ---------- Section 4: Categorical ----------
st.markdown("### 🏠 Loan Details & Lifestyle")
row4 = st.columns(3)

with row4[0]:
    residence_type = st.selectbox('🏡 Residence Type', ['Owned', 'Rented', 'Mortgage'])

with row4[1]:
    loan_purpose = st.selectbox('🎯 Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])

with row4[2]:
    loan_type = st.selectbox('🔐 Loan Type', ['Unsecured', 'Secured'])


# ---------- Prediction ----------
st.markdown("---")
st.markdown("### 🧮 Calculate Credit Risk")

if st.button('🔎 Calculate Credit Risk (PD & Score)', use_container_width=True):

    # Convert percentage inputs to decimal (VERY IMPORTANT)
    delinquency_ratio = delinquency_ratio / 100
    credit_utilization_ratio = credit_utilization_ratio / 100

    probability, credit_score, rating = predict(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency,
        delinquency_ratio,
        credit_utilization_ratio,
        num_open_accounts,
        residence_type,
        loan_purpose,
        loan_type
    )

    # ---------- Results ----------
    st.markdown("### 📊 Prediction Results")

    st.success(f"**🧨 Probability of Default (PD):** `{probability:.2%}`")
    st.info(f"**📊 Credit Score:** `{credit_score}`")
    st.write(f"**⭐ Rating:** `{rating}`")

    # ---------- Risk Interpretation ----------
    st.markdown("### ⚠️ Risk Assessment")

    if probability > 0.3:
        st.error("⚠️ High Risk: Customer likely to default")
    elif probability > 0.1:
        st.warning("⚠️ Medium Risk: Monitor closely")
    else:
        st.success("✅ Low Risk: Safe borrower")


# ---------- Footer ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)