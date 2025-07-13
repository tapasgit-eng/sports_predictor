from flask import Flask, render_template, request # type: ignore
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    return render_template('index.html', prediction_text=f'Predicted Runs: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
