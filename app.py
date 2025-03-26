from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    
    # Convert to DataFrame with correct feature names
    feature_names = ['feature1', 'feature2', 'feature3']  # Replace with your actual feature names
    df = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(df)
    
    return render_template('output.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)