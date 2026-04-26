# 🔍 Credit Risk Modeling & Scorecard System

This project implements an end-to-end credit risk assessment system to predict **Probability of Default (PD)** and generate a **scorecard-based credit score** using industry-standard techniques.

---

## 🚀 Key Features

- 📊 Predicts **Probability of Default (PD)** using Logistic Regression  
- 🧮 Converts PD into **Credit Score** using log-odds and PDO scaling  
- 📉 Incorporates real-world risk drivers:
  - Loan-to-Income Ratio
  - Delinquency Ratio (DPD-based)
  - Credit Utilization Ratio
- ⚖️ Handles class imbalance using SMOTE TOMEK  
- 📈 Evaluates model using:
  - ROC-AUC  
  - KS Statistic  
  - Gini Coefficient  
- 🌐 Interactive UI using Streamlit  

---

## 🧠 Credit Risk Framework

The system follows standard credit risk modeling principles:

- **PD (Probability of Default)** → Likelihood of borrower default  
- **Scorecard Transformation** → Converts PD into interpretable score  

### Score Formula:

Odds = (1 - PD) / PD  
Score = Offset + Factor × log(Odds)

Where:
- Factor = PDO / ln(2)  
- Offset = Base Score − Factor × ln(Base Odds)

---

## 🏗️ Project Architecture

User Input (Streamlit UI)  
        ↓  
Feature Engineering  
        ↓  
Logistic Regression Model (PD)  
        ↓  
Scorecard Transformation (PDO Scaling)  
        ↓  
Output:
  - PD  
  - Credit Score  
  - Risk Rating  

---

## 📊 Output

- **Probability of Default (PD)**  
- **Credit Score (300–900 range)**  
- **Risk Category (Low / Medium / High)**  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Pandas / NumPy  
- Streamlit  
- Joblib  

---

## ▶️ How to Run

pip install -r requirements.txt  
streamlit run main.py  

Then open:  
http://localhost:8501  

---

## 📌 Key Concepts Implemented

- Credit Risk Modeling  
- Scorecard Modeling (PDO, log-odds)  
- Feature Engineering (behavioral + financial)  
- Model Evaluation (KS, ROC, Gini)  
- Deployment (API + UI)  

---

## 💡 Future Improvements

- Add WOE-based scorecard  
- Time-based PD modeling  
- Model monitoring & drift detection  
- Integration with real credit bureau data  

---

## 👤 Author

Krushang Patel  

---

## ⭐ If you found this useful, give it a star!