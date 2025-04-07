import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def preprocess_symptoms(symptoms_text):
    """
    Preprocess symptom text for model input
    """
    # Convert to lowercase
    symptoms = symptoms_text.lower()
    
    # Remove extra whitespace
    symptoms = ' '.join(symptoms.split())
    
    # Remove special characters except commas
    symptoms = ''.join(char for char in symptoms if char.isalnum() or char in [',', ' '])
    
    return symptoms

def create_sample_data(n_samples=5000):
    # Age ranges and probabilities
    age_ranges = {
        'young': (18, 40),
        'middle': (41, 60),
        'elderly': (61, 90)
    }
    
    # Common symptoms that can appear in both conditions
    common_symptoms = [
        'fatigue',
        'dizziness',
        'nausea',
        'headache',
        'sweating'
    ]
    
    # Symptoms more indicative of heart disease
    heart_specific_symptoms = [
        'chest pain',
        'chest pressure',
        'arm pain',
        'jaw pain',
        'shortness of breath',
        'irregular heartbeat'
    ]
    
    # General symptoms
    general_symptoms = [
        'cough',
        'runny nose',
        'muscle tension',
        'trouble sleeping',
        'anxiety'
    ]
    
    data = []
    for _ in range(n_samples):
        # Age-based risk factors
        age_group = np.random.choice(['young', 'middle', 'elderly'], p=[0.3, 0.4, 0.3])
        age = np.random.randint(age_ranges[age_group][0], age_ranges[age_group][1])
        
        # Base disease probability affected by age
        base_prob = 0.1 if age_group == 'young' else 0.3 if age_group == 'middle' else 0.5
        
        # Add some randomness to disease probability
        disease_prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.05, 0.95)
        has_disease = np.random.choice([True, False], p=[disease_prob, 1-disease_prob])
        
        # Generate symptoms with noise
        num_symptoms = np.random.randint(2, 5)
        symptoms = []
        
        if has_disease:
            # Add 1-2 heart-specific symptoms
            symptoms.extend(np.random.choice(heart_specific_symptoms, 
                                          size=np.random.randint(1, 3), 
                                          replace=False))
        
        # Add 1-2 common symptoms
        symptoms.extend(np.random.choice(common_symptoms, 
                                      size=np.random.randint(1, 3), 
                                      replace=False))
        
        # Maybe add a general symptom
        if np.random.random() < 0.3:
            symptoms.append(np.random.choice(general_symptoms))
        
        # Shuffle and join symptoms
        np.random.shuffle(symptoms)
        symptom_text = ', '.join(symptoms[:num_symptoms])
        
        # Gender with slight risk variation
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        
        # Add noise to the disease label (misdiagnosis simulation)
        if np.random.random() < 0.05:  # 5% chance of label noise
            has_disease = not has_disease
        
        data.append({
            'age': age,
            'gender': gender,
            'symptoms': symptom_text,
            'disease': 'Heart Disease' if has_disease else 'Low Risk'
        })
    
    return pd.DataFrame(data)

def train_model():
    # Create dataset with more samples
    df = create_sample_data(n_samples=5000)
    
    # Initialize encoders
    gender_encoder = LabelEncoder()
    symptom_vectorizer = TfidfVectorizer(max_features=100)
    scaler = StandardScaler()
    
    # Encode features
    X_gender = gender_encoder.fit_transform(df['gender'])
    X_symptoms = symptom_vectorizer.fit_transform(df['symptoms'])
    X_age = scaler.fit_transform(df['age'].values.reshape(-1, 1))
    
    # Combine features
    X = np.hstack((X_age, X_gender.reshape(-1, 1), X_symptoms.toarray()))
    y = (df['disease'] == 'Heart Disease').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create feature pipeline
    feature_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier())
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_samples_split': [2, 5],
        'classifier__subsample': [0.8, 0.9, 1.0]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        feature_pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    print(f"Test set accuracy: {grid_search.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    train_model()