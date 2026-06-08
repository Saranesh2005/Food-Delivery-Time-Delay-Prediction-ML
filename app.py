from flask import Flask, render_template, request
import pandas as pd
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("final_delivery_dataset_realistic.csv")

# =========================
# DISTANCE CALCULATION
# =========================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

df["Distance_km"] = df.apply(
    lambda row: haversine(
        row["Restaurant_latitude"],
        row["Restaurant_longitude"],
        row["Delivery_location_latitude"],
        row["Delivery_location_longitude"]
    ),
    axis=1
)

# =========================
# ADD CONTEXTUAL FACTORS
# =========================

traffic_list = ["Low", "Medium", "High"]
weather_list = ["Clear", "Cloudy", "Rain"]
peak_list = ["Yes", "No"]
food_list = ["Pizza", "Snack", "Biryani", "Burger"]

df["Traffic"] = [
    random.choice(traffic_list)
    for _ in range(len(df))
]

df["Weather"] = [
    random.choice(weather_list)
    for _ in range(len(df))
]

df["Peak_Hour"] = [
    random.choice(peak_list)
    for _ in range(len(df))
]

df["Food_Type"] = [
    random.choice(food_list)
    for _ in range(len(df))
]

# =========================
# ENCODING
# =========================

le_weather = LabelEncoder()
le_traffic = LabelEncoder()
le_peak = LabelEncoder()
le_food = LabelEncoder()

df["Weather"] = le_weather.fit_transform(df["Weather"])
df["Traffic"] = le_traffic.fit_transform(df["Traffic"])
df["Peak_Hour"] = le_peak.fit_transform(df["Peak_Hour"])
df["Food_Type"] = le_food.fit_transform(df["Food_Type"])

# =========================
# FEATURES AND TARGET
# =========================

X = df[[
    "Distance_km",
    "Delivery_person_Ratings",
    "Weather",
    "Traffic",
    "Peak_Hour",
    "Food_Type"
]]

y = df["Time_taken(min)"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# RANDOM FOREST REGRESSION
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# DELAY RISK CLASSIFICATION
# =========================

def classify_delay(x):

    if x < 30:
        return 0

    elif x <= 45:
        return 1

    else:
        return 2

df["Delay_Risk"] = df["Time_taken(min)"].apply(
    classify_delay
)

X2 = X
y2 = df["Delay_Risk"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2,
    y2,
    test_size=0.2,
    random_state=42
)

risk_model = LogisticRegression(max_iter=1000)

risk_model.fit(X_train2, y_train2)

# =========================
# FLASK ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])

def home():

    eta = None
    risk = None

    if request.method == "POST":

        try:

            distance = float(request.form["distance"])
            rating = float(request.form["rating"])

            weather = request.form["weather"].title()
            traffic = request.form["traffic"].title()
            peak = request.form["peak"].title()
            food = request.form["food"].title()

            weather_encoded = le_weather.transform([weather])[0]
            traffic_encoded = le_traffic.transform([traffic])[0]
            peak_encoded = le_peak.transform([peak])[0]
            food_encoded = le_food.transform([food])[0]

            user_input = [[
                distance,
                rating,
                weather_encoded,
                traffic_encoded,
                peak_encoded,
                food_encoded
            ]]

            eta = round(model.predict(user_input)[0])

            risk_pred = risk_model.predict(user_input)[0]

            if risk_pred == 0:
                risk = "LOW"

            elif risk_pred == 1:
                risk = "MEDIUM"

            else:
                risk = "HIGH"

        except Exception as e:

            return f"Error: {e}"

    return render_template(
        "index.html",
        eta=eta,
        risk=risk
    )

# =========================
# RUN WEBSITE
# =========================

if __name__ == "__main__":

    app.run(debug=True)