from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

app = Flask(__name__)

model_cnn = tf.keras.models.load_model("cnn_model.h5")

iris = load_iris()
class_names = iris.target_names

@app.route('/predict', methods=['GET'])
def predict():
    try:
        features = []
        for i in range(1, 5):
            value = request.args.get(f'feature{i}')
            if value is None:
                return jsonify({"error": f"Missing parameter: feature{i}"}), 400
            features.append(float(value))  

        features = np.array(features).reshape(1, 1, 1, 4) / 255.0 

        prediction = model_cnn.predict(features)
        predicted_class_idx = int(np.argmax(prediction))  
        predicted_class_name = class_names[predicted_class_idx] 
        confidence = float(np.max(prediction)) 

        return jsonify({
            'prediction': predicted_class_name,  
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)