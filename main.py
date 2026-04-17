import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# ---------------------------
# CREATE FOLDERS
# ---------------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------
# STEP 1: GENERATE DATA
# ---------------------------
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "price": np.random.randint(100, 1000, n),
    "competitor_price": np.random.randint(100, 1000, n),
    "demand": np.random.randint(1, 100, n),
    "time_spent": np.random.randint(5, 100, n),
    "clicks": np.random.randint(1, 10, n),
    "season": np.random.choice(["summer", "winter", "festive"], n)
})

data.to_csv("data/raw/simulated_data.csv", index=False)
print("✅ Raw data created!")

# ---------------------------
# STEP 2: PREPROCESSING
# ---------------------------
df = pd.read_csv("data/raw/simulated_data.csv")

df.dropna(inplace=True)

df = pd.get_dummies(df, columns=["season"])

df["price_diff"] = df["price"] - df["competitor_price"]
df["demand_per_click"] = df["demand"] / (df["clicks"] + 1)
df["engagement"] = df["time_spent"] * df["clicks"]

df.to_csv("data/processed/cleaned_data.csv", index=False)
print("✅ Data preprocessed!")

# ---------------------------
# STEP 3: DEMAND MODEL
# ---------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop("demand", axis=1)
y = df["demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("📊 Model MSE:", mse)

with open("models/demand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")

sample = X_test.iloc[0:1]
pred = model.predict(sample)
print("🔍 Sample Prediction:", pred)

# ---------------------------
# STEP 4: RL PRICING
# ---------------------------
print("\n🚀 Starting RL Pricing Simulation...")

actions = [-50, 0, 50]
Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_state(row):
    return (int(row["price"] // 100), int(row["demand"] // 10))

def get_reward(price, demand):
    return price * demand

# ---------------------------
# 🔥 NEW: EMOTION FUNCTION
# ---------------------------
def emotion_factor(row):
    score = row["engagement"]

    if score > 300:
        return 1.2
    elif score < 100:
        return 0.8
    else:
        return 1.0

# ---------------------------
# 🔥 NEW: FAIR PRICING
# ---------------------------
def apply_fairness(original_price, new_price):
    max_increase = original_price * 0.2
    max_decrease = original_price * 0.3

    upper = original_price + max_increase
    lower = original_price - max_decrease

    return min(max(new_price, lower), upper)

# Train RL
df = df.sample(100)

for episode in range(3):
    print(f"⚡ Episode {episode+1} running...")

    for _, row in df.iterrows():

        state = get_state(row)

        if state not in Q:
            Q[state] = [0, 0, 0]

        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(actions))
        else:
            action_idx = np.argmax(Q[state])

        action = actions[action_idx]

        new_price = max(50, row["price"] + action)

        temp = row.copy()
        temp["price"] = new_price

        temp_df = pd.DataFrame([temp.drop("demand")])
        predicted_demand = model.predict(temp_df)[0]

        reward = get_reward(new_price, predicted_demand)

        next_state = (int(new_price // 100), int(predicted_demand // 10))

        if next_state not in Q:
            Q[next_state] = [0, 0, 0]

        Q[state][action_idx] = Q[state][action_idx] + alpha * (
            reward + gamma * max(Q[next_state]) - Q[state][action_idx]
        )

print("✅ RL Training Completed!")

# ---------------------------
# STEP 5: FINAL PRICING
# ---------------------------
test_row = df.iloc[0]
state = get_state(test_row)

best_action = actions[np.argmax(Q[state])]

# Base price from RL
base_price = test_row["price"] + best_action

# Apply emotion
factor = emotion_factor(test_row)
emotion_price = base_price * factor

# Apply fairness
recommended_price = apply_fairness(test_row["price"], emotion_price)

# ---------------------------
# STEP 6: EXPLAINABLE AI
# ---------------------------
def explain_price(row):
    reasons = []

    if row["demand"] > 70:
        reasons.append("High demand")
    elif row["demand"] < 30:
        reasons.append("Low demand")

    if row["engagement"] > 200:
        reasons.append("High user engagement")

    if row["price"] > row["competitor_price"]:
        reasons.append("Price higher than competitor")
    else:
        reasons.append("Competitive pricing")

    if factor > 1:
        reasons.append("High user interest detected")
    elif factor < 1:
        reasons.append("Low user interest detected")

    if recommended_price > row["price"]:
        reasons.append("Price increased within fair limits")
    else:
        reasons.append("Price adjusted for fairness")

    return ", ".join(reasons)

print("\n==============================")
print("💡 FINAL RESULT")
print("Original Price:", test_row["price"])
print("Recommended Price:", round(recommended_price, 2))
print("==============================")

print("📊 Reason:", explain_price(test_row))

input("\nPress Enter to exit...")