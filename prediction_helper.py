import numpy as np
import pandas as pd
import joblib

# ---------- Load Model and Components ----------
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale_']  # Ensure consistency with saved key


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

    # Input dictionary with encoding and dummy placeholders
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,

        # One-hot encoded categorical fields
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy/default values for features required by model
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

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Ensure all required columns for scaling exist
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0  # Fallback default if missing

    # Apply scaling
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Return DataFrame restricted to model features
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

    return calculate_credit_score(input_df)


# ---------- Credit Score Calculation ----------
def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """
    Computes credit score and probability using logistic regression output.
    """

    # Raw score from logistic model
    log_odds = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_prob = 1 / (1 + np.exp(-log_odds))  # Sigmoid

    non_default_prob = 1 - default_prob
    credit_score = base_score + non_default_prob.flatten() * scale_length

    # Credit rating assignment
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return float(default_prob.flatten()[0]), int(credit_score[0]), rating
