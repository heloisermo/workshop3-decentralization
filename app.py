import numpy as np
import pandas as pd
import requests
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

@app.route('/predict', methods=['GET'])
def predict():
    try:
        features = [float(request.args.get(f)) for f in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
        features = scaler.transform([features]).reshape(1, len(features), 1)
        prediction = model.predict(features)[0][0]
        return jsonify({"prediction": int(prediction > 0.5), "probability": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/consensus_predict', methods=['GET'])
def consensus_predict():
    try:
        api_urls = [
            "http://peer1-ngrok-url/predict",
            "http://peer2-ngrok-url/predict",
            "http://peer3-ngrok-url/predict"
        ]
        
        features = {f: request.args.get(f) for f in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]}
        predictions = []
        
        for url in api_urls:
            response = requests.get(url, params=features)
            if response.status_code == 200:
                predictions.append(response.json()["probability"])
        
        avg_prediction = np.mean(predictions) if predictions else 0
        return jsonify({"consensus_prediction": int(avg_prediction > 0.5), "average_probability": float(avg_prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
