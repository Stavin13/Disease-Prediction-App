import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
import datetime
import joblib
from model import train_model, preprocess_symptoms

# Load API Key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load models and encoders
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('disease_model.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        symptom_vectorizer = joblib.load('symptom_vectorizer.joblib')
    except FileNotFoundError:
        print("Training new model...")
        train_model()
        model = joblib.load('disease_model.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        symptom_vectorizer = joblib.load('symptom_vectorizer.joblib')
    return model, gender_encoder, symptom_vectorizer

# Update prediction function
def predict_disease(age, gender, symptoms):
    model, gender_encoder, symptom_vectorizer = load_ml_components()
    
    try:
        # Process features
        gender_encoded = gender_encoder.transform([gender])
        symptoms_vectorized = symptom_vectorizer.transform([symptoms])
        age_formatted = np.array([age]).reshape(-1, 1)
        
        # Combine features
        X = np.hstack((age_formatted, gender_encoded.reshape(-1, 1), 
                      symptoms_vectorized.toarray()))
        
        # Get prediction and probability
        risk_score = int(model.predict_proba(X)[0][1] * 100)
        disease = "Heart Disease" if risk_score > 50 else "Low Risk"
        
        return disease, risk_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error in prediction", 0

# Query OpenRouter with AI models
def query_openrouter(prompt, models=["reka-core", "deepseek-chat", "mistral-7b-instruct"]):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "Disease Predictor"
    }
    for model in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": [{"role": "user", "content": prompt}]}
            )
            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"], model
        except Exception as e:
            print(f"Model {model} failed: {e}")
    return "Unable to explain the result.", "None"

def get_ai_explanation(age, gender, symptoms, prediction, risk_score):
    """Get AI explanation for the prediction"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost:8501",
        "Content-Type": "application/json"
    }
    
    # Prepare prompt for the AI
    prompt = f"""
    As a medical AI assistant, explain this heart disease risk assessment:
    Patient Details:
    - Age: {age}
    - Gender: {gender}
    - Symptoms: {symptoms}
    
    Prediction: {prediction}
    Risk Score: {risk_score}%
    
    Provide a brief, clear explanation of this assessment and what it means for the patient.
    """
    
    # Models to try in order of preference
    models = [
        "mistralai/mistral-7b-instruct",
        "anthropic/claude-2",
        "google/palm-2-chat-bison"
    ]
    
    for model in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                explanation = response.json()["choices"][0]["message"]["content"]
                return explanation
        except Exception as e:
            continue
    
    return "Unable to generate AI explanation at this moment."

def add_to_history(name, age, gender, symptoms, disease, risk_score):
    """Add prediction to history"""
    st.session_state.history.append({
        "Name": name,
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Disease": disease,
        "Score": risk_score,
        "Date": str(datetime.date.today())
    })

def show_prediction_history():
    """Display prediction history"""
    if st.session_state.history:
        st.markdown("### 📜 Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title with emoji
    st.title("🏥 Disease Risk Prediction")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("👤 Name")
        age = st.slider("🎯 Age", 1, 100, 25)
    
    with col2:
        gender = st.radio("⚥ Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("🔍 Symptoms (comma-separated)")
    
    # Center the predict button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("🔮 Predict")
    
    if predict_button:
        if name and symptoms:
            # Get prediction
            disease, risk_score = predict_disease(age, gender, symptoms)
            
            # Get AI explanation
            explanation = get_ai_explanation(age, gender, symptoms, disease, risk_score)
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"Disease: {disease}")
            st.write(f"Risk Score: {risk_score}%")
            
            st.subheader("AI Explanation")
            st.write(explanation)
            
            # Add to history
            add_to_history(name, age, gender, symptoms, disease, risk_score)
            
            # Show history
            show_prediction_history()
        else:
            st.warning("Please fill in all fields")

if __name__ == "__main__":
    main()
