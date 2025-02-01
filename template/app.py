from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('crop_recommender.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['n'],
        data['p'],
        data['k'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    crop = label_encoder.inverse_transform(prediction)[0]
    return jsonify({'crop': crop})

if __name__ == '__main__':
    app.run(debug=True)
    