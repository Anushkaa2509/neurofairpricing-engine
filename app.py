import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from database import create_table, validate_user, add_user, save_history, get_history

# ---------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------
st.set_page_config(layout="wide")

# ---------------------------
# CREATE TABLE
# ---------------------------
create_table()

# ---------------------------
# LOGIN SYSTEM
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login / Signup")

    option = st.selectbox("Select", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            add_user(username, password)
            st.success("Account created! Now login.")

    if option == "Login":
        if st.button("Login"):
            if validate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------------------
# LOAD MODEL (DEPLOY SAFE)
# ---------------------------
model_path = "models/demand_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload 'models/demand_model.pkl'")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# ---------------------------
# MAIN UI
# ---------------------------
st.title("💰 NeuroFairPricing Engine")
st.write("### 🔧 Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.slider("Current Price", 100, 1000, 500)
    competitor_price = st.slider("Competitor Price", 100, 1000, 500)
    season = st.selectbox("Season", ["summer", "winter", "festive"])

with col2:
    time_spent = st.slider("Time Spent (seconds)", 5, 100, 50)
    clicks = st.slider("Clicks", 1, 10, 5)

# ---------------------------
# BUTTON (IMPORTANT)
# ---------------------------
if st.button("🚀 Predict Optimal Price"):

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    engagement = time_spent * clicks
    price_diff = price - competitor_price

    season_summer = 1 if season == "summer" else 0
    season_winter = 1 if season == "winter" else 0
    season_festive = 1 if season == "festive" else 0

    input_df = pd.DataFrame([{
        "price": price,
        "competitor_price": competitor_price,
        "time_spent": time_spent,
        "clicks": clicks,
        "season_festive": season_festive,
        "season_summer": season_summer,
        "season_winter": season_winter,
        "price_diff": price_diff,
        "demand_per_click": 0,
        "engagement": engagement
    }])

    # ---------------------------
    # DEMAND PREDICTION
    # ---------------------------
    predicted_demand = model.predict(input_df)[0]

    # ---------------------------
    # RL LOGIC (SIMPLIFIED)
    # ---------------------------
    if predicted_demand > 70:
        action = 50
    elif predicted_demand < 30:
        action = -50
    else:
        action = 0

    base_price = price + action

    # ---------------------------
    # EMOTION AI
    # ---------------------------
    def emotion_factor(eng):
        if eng > 300:
            return 1.2
        elif eng < 100:
            return 0.8
        return 1.0

    factor = emotion_factor(engagement)
    emotion_price = base_price * factor

    # ---------------------------
    # FAIR PRICING
    # ---------------------------
    def apply_fairness(original, new):
        upper = original * 1.2
        lower = original * 0.7
        return min(max(new, lower), upper)

    recommended_price = apply_fairness(price, emotion_price)

    # ---------------------------
    # EXPLAINABLE AI
    # ---------------------------
    reasons = []

    if predicted_demand > 70:
        reasons.append("High demand")
    elif predicted_demand < 30:
        reasons.append("Low demand")

    if engagement > 200:
        reasons.append("High engagement")

    if price > competitor_price:
        reasons.append("Higher than competitor")
    else:
        reasons.append("Competitive pricing")

    if factor > 1:
        reasons.append("High user interest")
    elif factor < 1:
        reasons.append("Low user interest")

    # ---------------------------
    # SAVE HISTORY (ONLY ON CLICK)
    # ---------------------------
    save_history(
        st.session_state.username,
        price,
        recommended_price,
        predicted_demand,
        ", ".join(reasons)
    )

    # ---------------------------
    # OUTPUT
    # ---------------------------
    st.write("---")
    st.subheader("📊 Results")

    col3, col4, col5 = st.columns(3)

    col3.metric("Demand", round(predicted_demand, 2))
    col4.metric("Original Price", price)
    col5.metric("Recommended Price", round(recommended_price, 2))

    # ---------------------------
    # GRAPH
    # ---------------------------
    st.write("### 📈 Demand vs Price")

    price_range = np.linspace(100, 1000, 50)
    demand_preds = []

    for p in price_range:
        temp = input_df.copy()
        temp["price"] = p
        temp["price_diff"] = p - competitor_price
        demand_preds.append(model.predict(temp)[0])

    fig, ax = plt.subplots()
    ax.plot(price_range, demand_preds)
    ax.set_xlabel("Price")
    ax.set_ylabel("Demand")

    st.pyplot(fig)

    # ---------------------------
    # EXPLANATION
    # ---------------------------
    st.write("### 🧠 Explanation")
    st.write(", ".join(reasons))

# ---------------------------
# HISTORY
# ---------------------------
st.write("### 📜 Pricing History")

history = get_history(st.session_state.username)

if history:
    df_history = pd.DataFrame(history, columns=[
        "ID", "User", "Original Price", "Recommended Price", "Demand", "Reason"
    ])
    st.dataframe(df_history)
else:
    st.write("No history yet")
