import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_model():
    # Load Excel data
    df = pd.read_excel('mental_health_data.xlsx')
    
    # Define features and target
    X = df.drop('Depression', axis=1)
    y = df['Depression']
    
    # Preprocessing pipeline
    categorical_features = ['Gender', 'City', 'Profession', 'Degree']
    numerical_features = ['Age', 'CGPA', 'Sleep Duration', 'Work/Study Hours']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'model.pkl')
    print("Model trained and saved!")

if __name__ == '__main__':
    train_model()