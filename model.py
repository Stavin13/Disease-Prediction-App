import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def preprocess_symptoms(symptoms_text):
    """Preprocess symptom text for model input"""
    return symptoms_text.lower().split(',')

def train_model():
    """Train the disease prediction model"""
    # Create sample dataset
    df = create_sample_data()
    
    # Initialize encoders
    gender_encoder = LabelEncoder()
    symptom_vectorizer = TfidfVectorizer(max_features=100)
    age_scaler = StandardScaler()
    
    # Encode features
    X_gender = gender_encoder.fit_transform(df['gender'])
    X_symptoms = symptom_vectorizer.fit_transform(df['symptoms'].astype(str))
    X_age = age_scaler.fit_transform(df['age'].values.reshape(-1, 1))
    
    # Combine features
    X = np.hstack((X_age, X_gender.reshape(-1, 1), X_symptoms.toarray()))
    y = (df['disease'] == 'Heart Disease').astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save components
    joblib.dump(model, 'disease_model.joblib')
    joblib.dump(gender_encoder, 'gender_encoder.joblib')
    joblib.dump(symptom_vectorizer, 'symptom_vectorizer.joblib')
    joblib.dump(age_scaler, 'age_scaler.joblib')
    
    return True

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.gender_encoder = None
        self.symptom_vectorizer = None
        self.age_scaler = None
        self.load_components()
    
    def load_components(self):
        try:
            self.model = joblib.load('disease_model.joblib')
            self.gender_encoder = joblib.load('gender_encoder.joblib')
            self.symptom_vectorizer = joblib.load('symptom_vectorizer.joblib')
            self.age_scaler = joblib.load('age_scaler.joblib')
        except FileNotFoundError:
            train_model()
            self.load_components()
    
    def predict(self, age, gender, symptoms):
        try:
            # Process features
            gender_encoded = self.gender_encoder.transform([gender])
            symptoms_vectorized = self.symptom_vectorizer.transform([symptoms])
            age_scaled = self.age_scaler.transform([[age]])
            
            # Combine features
            X = np.hstack((age_scaled, 
                          gender_encoded.reshape(-1, 1), 
                          symptoms_vectorized.toarray()))
            
            # Get prediction and probability
            risk_score = int(self.model.predict_proba(X)[0][1] * 100)
            return risk_score
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

def create_sample_data(n_samples=1000):
    """Create synthetic training data"""
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'symptoms': [
            ', '.join(np.random.choice([
                'chest pain', 'shortness of breath', 'fatigue',
                'nausea', 'dizziness', 'cold sweat', 'headache'
            ], np.random.randint(1, 4))) for _ in range(n_samples)
        ],
        'disease': np.random.choice(['Heart Disease', 'Low Risk'], n_samples)
    }
    return pd.DataFrame(data)