import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
import datetime
import joblib
from model import train_model, preprocess_symptoms
import plotly.express as px
import plotly.graph_objects as go

# Add this near the top of the file after imports
if 'history' not in st.session_state:
    st.session_state.history = []

# Load API Key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Add debug code to check API key
def check_api_key():
    if not OPENROUTER_API_KEY:
        st.sidebar.error("⚠️ API Key Missing!")
        st.sidebar.markdown("""
        1. Check if `.env` file exists
        2. Verify content: `OPENROUTER_API_KEY=your-key`
        3. Key format should start with 'sk-or-v1-'
        """)
        return False
    else:
        st.sidebar.success("✅ API Key Loaded")
        # Show masked key for verification
        masked_key = f"...{OPENROUTER_API_KEY[-8:]}"
        st.sidebar.info(f"Key ends with: {masked_key}")
        return True

# Load models and encoders
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('disease_model.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        symptom_vectorizer = joblib.load('symptom_vectorizer.joblib')
        age_scaler = joblib.load('age_scaler.joblib')
        return model, gender_encoder, symptom_vectorizer, age_scaler
    except FileNotFoundError:
        st.warning("Training new model... This may take a moment.")
        try:
            train_model()  # Train the model
            # Try loading again after training
            model = joblib.load('disease_model.joblib')
            gender_encoder = joblib.load('gender_encoder.joblib')
            symptom_vectorizer = joblib.load('symptom_vectorizer.joblib')
            age_scaler = joblib.load('age_scaler.joblib')
            return model, gender_encoder, symptom_vectorizer, age_scaler
        except Exception as e:
            st.error(f"Error in model training/loading: {str(e)}")
            raise

# Update prediction function
def predict_disease(age, gender, symptoms, severity="Moderate", duration="Recent (Days)"):
    """Enhanced disease prediction with severity and duration"""
    model, gender_encoder, symptom_vectorizer, age_scaler = load_ml_components()
    
    try:
        # Process features
        gender_encoded = gender_encoder.transform([gender])
        symptoms_vectorized = symptom_vectorizer.transform([symptoms])
        age_scaled = age_scaler.transform(np.array([age]).reshape(-1, 1))
        
        # Add severity and duration factors
        severity_factor = {"Mild": 0.8, "Moderate": 1.0, "Severe": 1.2}
        duration_factor = {
            "Recent (Days)": 0.9,
            "Short-term (Weeks)": 1.0,
            "Long-term (Months+)": 1.2
        }
        
        # Combine features
        X = np.hstack((age_scaled, gender_encoded.reshape(-1, 1), 
                      symptoms_vectorized.toarray()))
        
        # Get prediction and adjust by severity/duration
        base_risk_score = model.predict_proba(X)[0][1] * 100
        adjusted_score = base_risk_score * severity_factor[severity] * duration_factor[duration]
        adjusted_score = min(100, adjusted_score)  # Cap at 100
        
        # Expanded disease classification
        if adjusted_score > 75:
            disease = "High Risk - Immediate Attention Needed"
        elif adjusted_score > 50:
            disease = "Moderate Risk - Medical Consultation Recommended"
        else:
            disease = "Low Risk - Monitor Symptoms"
        
        return disease, int(adjusted_score)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error in prediction", 0

# Query OpenRouter with AI models
def query_openrouter(prompt):
    """Query OpenRouter API for AI explanation"""
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API key not found. Please check your .env file.")
        return "Unable to generate explanation: API key missing."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Provide clear, professional explanations of disease risk assessments."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_ai_explanation(age, gender, symptoms, prediction, risk_score):
    """Get AI explanation for the prediction"""
    prompt = f"""
    Analyze this heart disease risk assessment and provide a clear medical explanation:

    Patient Profile:
    - Age: {age}
    - Gender: {gender}
    - Reported Symptoms: {symptoms}

    Assessment Results:
    - Prediction: {prediction}
    - Risk Score: {risk_score}%

    Please provide:
    1. A brief interpretation of the risk score
    2. Key factors that influenced this prediction
    3. General health recommendations (avoid specific medical advice)
    4. When to seek medical attention
    """

    explanation = query_openrouter(prompt)
    
    if not explanation:
        return "Unable to generate explanation. Please try again."
        
    return explanation

def add_to_history(name, age, gender, symptoms, disease, risk_score):
    """Add prediction to history"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        'Timestamp': timestamp,
        'Name': name,
        'Age': age,
        'Gender': gender,
        'Symptoms': symptoms,
        'Disease': disease,
        'Score': risk_score
    })

def show_prediction_history():
    """Display prediction history in a table"""
    if st.session_state.history:
        st.markdown("### 📋 Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

def get_symptom_suggestions():
    """Get comprehensive symptom suggestions by category"""
    return {
        "Cardiovascular": [
            "chest pain", "shortness of breath", "irregular heartbeat",
            "rapid heartbeat", "slow heartbeat", "dizziness when standing",
            "fainting", "swollen legs", "cold extremities", "chest pressure"
        ],
        "Respiratory": [
            "persistent cough", "wheezing", "coughing up blood",
            "difficulty breathing", "rapid breathing", "chest congestion",
            "sleep apnea", "excessive sputum", "noisy breathing"
        ],
        "Neurological": [
            "severe headache", "confusion", "memory problems",
            "difficulty speaking", "vision changes", "weakness in limbs",
            "tremors", "loss of balance", "seizures", "numbness"
        ],
        "Gastrointestinal": [
            "severe abdominal pain", "persistent nausea", "vomiting",
            "difficulty swallowing", "unexplained weight loss",
            "blood in stool", "heartburn", "loss of appetite"
        ],
        "General": [
            "fatigue", "fever", "night sweats", "unexplained weight loss",
            "muscle weakness", "joint pain", "skin changes", "anxiety",
            "depression", "sleep problems"
        ]
    }

def validate_symptoms(symptoms):
    """Validate symptoms against expanded list"""
    all_symptoms = []
    for category in get_symptom_suggestions().values():
        all_symptoms.extend(category)
    
    symptom_list = [s.strip().lower() for s in symptoms.split(",")]
    valid_symptoms = [s for s in symptom_list if s in all_symptoms]
    unknown_symptoms = [s for s in symptom_list if s not in all_symptoms]
    return valid_symptoms, unknown_symptoms

def display_symptom_guide():
    """Display organized symptom categories"""
    st.markdown("### 🏥 Symptom Guide")
    symptoms_dict = get_symptom_suggestions()
    
    for category, symptoms in symptoms_dict.items():
        with st.expander(f"📌 {category} Symptoms"):
            st.write(", ".join(symptoms))

def show_analytics():
    """Display analytics dashboard"""
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("### 📊 Analytics Dashboard")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Age Analysis", "Gender Distribution"])
        
        with tab1:
            fig = px.histogram(df, x="Score", 
                             title="Risk Score Distribution",
                             color="Disease")
            st.plotly_chart(fig)
        
        with tab2:
            fig = px.scatter(df, x="Age", y="Score",
                           color="Disease",
                           title="Age vs Risk Score")
            st.plotly_chart(fig)
        
        with tab3:
            fig = px.pie(df, names="Gender", 
                        title="Gender Distribution")
            st.plotly_chart(fig)

def export_data():
    """Export prediction data"""
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Export options
        export_format = st.selectbox("Select Format", 
                                   ["CSV", "Excel", "JSON"])
        
        if export_format == "CSV":
            data = df.to_csv(index=False)
            mime = "text/csv"
            file_ext = "csv"
        elif export_format == "Excel":
            data = df.to_excel(index=False)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_ext = "xlsx"
        else:
            data = df.to_json(orient="records")
            mime = "application/json"
            file_ext = "json"
            
        st.download_button(
            label=f"Download {export_format}",
            data=data,
            file_name=f"predictions.{file_ext}",
            mime=mime
        )

def show_prediction_tool():
    """Show the prediction tool interface"""
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
    
    # Replace simple symptom display with organized categories
    display_symptom_guide()
    
    # Add severity selection
    severity = st.select_slider(
        "Symptom Severity",
        options=["Mild", "Moderate", "Severe"],
        value="Moderate"
    )
    
    # Add duration
    duration = st.select_slider(
        "Duration of Symptoms",
        options=["Recent (Days)", "Short-term (Weeks)", "Long-term (Months+)"],
        value="Recent (Days)"
    )
    
    # Center the predict button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Predict"):
            if name and symptoms:
                with st.spinner("Analyzing symptoms..."):
                    disease, risk_score = predict_disease(age, gender, symptoms, severity, duration)
                    
                    st.subheader("🏥 Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Disease Prediction", disease)
                    with col2:
                        st.metric("Risk Score", f"{risk_score}%")
                    
                    # Debug section
                    st.sidebar.markdown("### 🔍 Debug Info")
                    if st.sidebar.checkbox("Show API Details"):
                        st.sidebar.write("API Key Status:", "✅ Present" if OPENROUTER_API_KEY else "❌ Missing")
                    
                    with st.spinner("Generating medical explanation..."):
                        explanation = get_ai_explanation(age, gender, symptoms, disease, risk_score)
                        if explanation != "Unable to generate explanation. Please try again.":
                            st.markdown("### 🤖 AI Medical Explanation")
                            st.markdown(explanation)
                    
                    add_to_history(name, age, gender, symptoms, disease, risk_score)
                    show_prediction_history()
                    
                    # Add analytics
                    show_analytics()
                    
                    # Add export option
                    st.markdown("### 📥 Export Data")
                    export_data()
            else:
                st.warning("Please fill in all required fields")

def show_admin_dashboard():
    """Show the admin dashboard interface"""
    st.markdown("### 🛠️ Admin Dashboard")
    st.write("Admin functionalities will be implemented here.")

def login_user():
    """Login user interface"""
    st.markdown("### 🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.session_state.is_admin = True
        elif username and password:
            st.session_state.logged_in = True
            st.session_state.is_admin = False
        else:
            st.warning("Invalid credentials")

def main():
    if 'logged_in' not in st.session_state:
        login_user()
        return
    
    # Add API key check
    api_key_valid = check_api_key()
    
    if st.session_state.get('is_admin'):
        tab1, tab2 = st.tabs(["Prediction Tool", "Admin Dashboard"])
        with tab1:
            show_prediction_tool()
        with tab2:
            show_admin_dashboard()
    else:
        show_prediction_tool()

if __name__ == "__main__":
    main()
