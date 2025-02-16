import requests
import pandas as pd
from collections import Counter

#We firstly used ngrok to create link available for each other but to continue the work in our side we use local links
model_endpoints = [
    "http://10.5.0.2:5001/predict",  # CNN
    "http://10.5.0.2:5002/predict",  # DNN
    "http://10.5.0.2:5003/predict",  # LSTM
    "http://10.5.0.2:5004/predict",  # RNN
    "http://10.5.0.2:5005/predict"   # Transformer
]

def get_predictions(features):
    predictions = []
    for url in model_endpoints:
        try:
            response = requests.get(url, params=features)
            result = response.json()

            if "prediction" in result:
                prediction_value = result["prediction"]
                print(f"Model at {url} predicted {prediction_value}")
                predictions.append(prediction_value)  
        except Exception as e:
            print(f"Error contacting {url}: {e}")
    
    return predictions

def consensus_prediction(features):
    predictions = get_predictions(features)   
    if predictions:
        class_count = Counter(predictions)
        predicted_class = class_count.most_common(1)[0][0] 
        return predicted_class
    return None

df = pd.read_json('dataset.json')

consensus_predictions = []

for index, row in df.iterrows():
    print("\nIteration n°",index+1," on ", len(df))
    features = {
        'feature1': row['sepalLength'],
        'feature2': row['sepalWidth'],
        'feature3': row['petalLength'],
        'feature4': row['petalWidth']
    }
    
    prediction = consensus_prediction(features)
    
    consensus_predictions.append(prediction)

print("Consensus Predictions for all rows:")
print(consensus_predictions)