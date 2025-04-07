# utils/health_tips.py
def get_health_tips(disease, risk):
    tips = {
        "Heart Disease": {
            "High": [
                "Avoid saturated fats and reduce salt intake.",
                "Exercise at least 30 minutes daily.",
                "Schedule regular checkups with your cardiologist."
            ],
            "Moderate": [
                "Maintain a healthy diet and monitor cholesterol.",
                "Incorporate walking or cycling into your routine."
            ],
            "Low": [
                "Stay active and keep stress levels low.",
                "Continue regular health screenings."
            ]
        }
    }
    return tips.get(disease, {}).get(risk, ["Stay healthy!"])
