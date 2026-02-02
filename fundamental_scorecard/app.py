import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title("Fundamental Scorecard + ML")

# Load CSV
df = pd.read_csv("raw_data.csv")

# Remove Google Sheets junk column
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

# --- Ratios ---
df["ROE"] = df["Net Profit"] / df["Equity"]
df["ROA"] = df["Net Profit"] / df["Total Assets"]
df["Margin"] = df["Net Profit"] / df["Revenue"]
df["DebtEquity"] = df["Total Debt"] / df["Equity"]

# --- Percentile Scoring (USP) ---
df["ROE_pct"] = df["ROE"].rank(pct=True)
df["ROA_pct"] = df["ROA"].rank(pct=True)
df["Margin_pct"] = df["Margin"].rank(pct=True)

df["Final_Score"] = (
    df["ROE_pct"]*0.4 +
    df["ROA_pct"]*0.3 +
    df["Margin_pct"]*0.3
) * 100

df = df.sort_values("Final_Score", ascending=False)

# --- Rule-based Labels (for ML training) ---
df["Signal"] = "Hold"
df.loc[df["Final_Score"] >= 70, "Signal"] = "Buy"
df.loc[df["Final_Score"] < 40, "Signal"] = "Avoid"

# --- Machine Learning ---
features = df[["ROE_pct","ROA_pct","Margin_pct","DebtEquity"]]
labels = df["Signal"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, labels)

df["ML_Prediction"] = model.predict(features)

# --- Dashboard ---

st.subheader("ML Enhanced Fundamental Scorecard")

st.dataframe(df[[
    "Company",
    "Sector",
    "Final_Score",
    "Signal",
    "ML_Prediction"
]])

st.subheader("Final Score Ranking")
st.bar_chart(df.set_index("Company")["Final_Score"])

# Company selector
company = st.selectbox("Select Company", df["Company"])

st.subheader("Selected Company Details")
st.dataframe(df[df["Company"] == company])
