from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Home route (Input Form)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            experience = float(request.form["experience"])  # Get user input
            prediction = model.predict(np.array([[experience]]))  # Make prediction
            predicted_salary = round(prediction[0], 2)  # Round to 2 decimal places
            return render_template("output.html", experience=experience, salary=predicted_salary)
        except ValueError:
            return "Invalid input! Please enter a numeric value."
    
    return render_template("input.html")

if __name__ == "__main__":
    app.run(debug=True)
