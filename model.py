import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.diseases = ['Heart Disease', 'Diabetes', 'Lung Cancer']
    
    def create_sample_data(self, disease_type):
        if disease_type == 'Heart Disease':
            return self._create_heart_data()
        elif disease_type == 'Diabetes':
            return self._create_diabetes_data()
        else:
            return self._create_lung_cancer_data()
    
    def train_all_models(self):
        for disease in self.diseases:
            X, y = self.create_sample_data(disease)
            self._train_model(disease, X, y)
    
    def _train_model(self, disease, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models[disease] = model
        self.scalers[disease] = scaler
        
        # Save to disk
        joblib.dump(model, f'{disease.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(scaler, f'{disease.lower().replace(" ", "_")}_scaler.joblib')