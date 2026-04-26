import numpy as np
import pandas as pd
import joblib

# ---------- Load Model and Components ----------
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale_']


# ---------- Prepare Input Data ----------
def prepare_input(
    age, income, loan_amount, loan_tenure_months,
    avg_dpd_per_delinquency, delinquency_ratio, credit_utilization_ratio,
    num_open_accounts, residence_type, loan_purpose, loan_type
):
    """
    Builds and scales input DataFrame based on user input.
    """

    # Derived feature
    loan_to_income = loan_amount / income if income > 0 else 0

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,

        # One-hot encoding
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Default placeholders
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Ensure all scaling columns exist
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure all model features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


# ---------- Prediction Interface ----------
def predict(
    age, income, loan_amount, loan_tenure_months,
    avg_dpd_per_delinquency, delinquency_ratio, credit_utilization_ratio,
    num_open_accounts, residence_type, loan_purpose, loan_type
):
    """
    Handles prediction flow using user inputs.
    """

    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, delinquency_ratio,
        credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    return calculate_credit_score(input_df, model)


# ---------- Credit Score Calculation ----------
def calculate_credit_score(input_df, model,
                           base_score=600,
                           base_odds=50,
                           pdo=20):
    """
    Converts logistic regression output into scorecard-style credit score.
    """

    # Step 1: Log odds
    log_odds = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Step 2: Probability of Default (PD)
    pd = 1 / (1 + np.exp(-log_odds))

    # Safety clipping (VERY IMPORTANT)
    pd = np.clip(pd, 1e-6, 1 - 1e-6)

    # Step 3: Convert PD → Odds
    odds = (1 - pd) / pd

    # Step 4: PDO Scaling
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    # Step 5: Final Score
    credit_score = offset + factor * np.log(odds)

    # Step 6: Rating
    def get_rating(score):
        if score < 500:
            return 'Poor'
        elif score < 650:
            return 'Average'
        elif score < 750:
            return 'Good'
        else:
            return 'Excellent'

    rating = get_rating(credit_score[0])

    return float(pd.flatten()[0]), int(credit_score[0]), rating