# Disease Prediction System

A machine learning-powered web application that predicts disease risks based on patient symptoms and characteristics.

## 🌟 Features

- Multiple disease prediction (Heart Disease, Diabetes, Lung Cancer)
- AI-powered explanations for predictions
- User authentication and history tracking
- Interactive visualizations and analytics
- Mobile-responsive interface

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Models**: scikit-learn
- **AI Integration**: OpenRouter API
- **Database**: SQLite
- **Visualization**: Plotly

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- OpenRouter API key

## 🛠️ Installation

1. Clone the repository:
```powershell
git clone https://github.com/YourUsername/disease-prediction-app.git
cd disease-prediction-app
```

2. Create virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENROUTER_API_KEY=your-api-key-here
```

## 🏃‍♂️ Running the App

```powershell
python -m streamlit run app.py
```

Access the app at: https://diseasepredapp.streamlit.app/

## 📱 Usage

1. Login with default credentials:
   - Username: admin
   - Password: admin

2. Select disease type for prediction

3. Enter patient information:
   - Age
   - Gender
   - Symptoms

4. Get predictions with AI explanations

5. View history and analytics

## 📊 Features in Detail

### Disease Prediction
- Heart Disease Risk Assessment
- Diabetes Risk Prediction
- Lung Cancer Risk Analysis

### Analytics
- Risk Score Timeline
- Disease Distribution
- Patient Demographics

### User Management
- Secure Authentication
- History Tracking
- Admin Dashboard

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- OpenRouter API for AI capabilities
- Streamlit for the web framework
- scikit-learn for ML models