import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset.csv")

# Features (X) and target variable (y)
X = data[['Experience']]  # Independent variable
y = data['Salary']  # Dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved successfully!")
