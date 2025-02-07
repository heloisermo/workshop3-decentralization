import numpy as np
import pandas as pd
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from flask import Flask, request, jsonify

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
X = df.drop("Survived", axis=1).values
y = df["Survived"].values

test_size = 0.2
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = Sequential([
    SimpleRNN(16, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

model.save("titanic_rnn.h5")

app = Flask(__name__)

BALANCE_FILE = "balances.json"

try:
    with open(BALANCE_FILE, "r") as f:
        model_balances = json.load(f)
except FileNotFoundError:
    model_balances = {"peer1": 1000, "peer2": 1000, "peer3": 1000}
    with open(BALANCE_FILE, "w") as f:
        json.dump(model_balances, f)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        features = [float(request.args.get(f)) for f in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
        features = scaler.transform([features]).reshape(1, len(features), 1)
        prediction = model.predict(features)[0][0]
        return jsonify({"prediction": int(prediction > 0.5), "probability": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

model_weights = {"peer1": 1.0, "peer2": 1.0, "peer3": 1.0}  

def update_weights_and_balance(predictions, consensus):
    global model_weights, model_balances
    for peer, pred in predictions.items():
        error = abs(pred - consensus)
        penalty = error * 10 
        model_weights[peer] = max(0, model_weights[peer] - error * 0.1)
        model_balances[peer] = max(0, model_balances[peer] - penalty)
    with open(BALANCE_FILE, "w") as f:
        json.dump(model_balances, f)

@app.route('/consensus_predict', methods=['GET'])
def consensus_predict():
    try:
        api_urls = {
            "peer1": "http://peer1-ngrok-url/predict",
            "peer2": "http://peer2-ngrok-url/predict",
            "peer3": "http://peer3-ngrok-url/predict"
        }
        
        features = {f: request.args.get(f) for f in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]}
        predictions = {}
        weighted_sum = 0
        weight_total = 0
        
        for peer, url in api_urls.items():
            response = requests.get(url, params=features)
            if response.status_code == 200:
                prob = response.json()["probability"]
                predictions[peer] = prob
                weighted_sum += prob * model_weights[peer]
                weight_total += model_weights[peer]
        
        consensus_prediction = weighted_sum / weight_total if weight_total > 0 else 0
        update_weights_and_balance(predictions, consensus_prediction)
        
        return jsonify({
            "consensus_prediction": int(consensus_prediction > 0.5),
            "average_probability": float(consensus_prediction),
            "model_weights": model_weights,
            "model_balances": model_balances
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
