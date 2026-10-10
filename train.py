import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 1. Provide initial sample gym profiles
data = {
    "tier": [
        "Metro",
        "Metro",
        "Tier-2",
        "Tier-2",
        "Suburban",
        "Metro",
        "Suburban",
    ],
    "area_sqft": [6000, 2500, 4000, 2000, 1500, 10000, 1800],
    "has_pool": [1, 0, 0, 0, 0, 1, 0],
    "has_sauna": [1, 0, 1, 0, 0, 1, 0],
    "monthly_rate": [5500, 2200, 1800, 1400, 1000, 7000, 1100],
}
df = pd.DataFrame(data)

# 2. Separate input features (X) and target price (y)
X = df[["tier", "area_sqft", "has_pool", "has_sauna"]]
y = df["monthly_rate"]

# 3. Convert words ('Metro') into numbers the AI can read
X_encoded = pd.get_dummies(X, columns=["tier"])

# 4. Train the Random Forest algorithm
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_encoded, y)

# 5. Save the trained brain and column structure to a file
joblib.dump(
    {"model": model, "columns": X_encoded.columns.tolist()}, "gym_brain.pkl"
)
print("Success! Trained model saved as gym_brain.pkl")