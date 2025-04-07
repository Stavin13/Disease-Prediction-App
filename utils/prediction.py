# utils/prediction.py
import random

def predict_disease(inputs):
    # Mock logic — replace with actual ML model
    score = random.randint(0, 100)
    if score > 80:
        risk = "High"
    elif score > 50:
        risk = "Moderate"
    else:
        risk = "Low"
    return {
        "disease": "Heart Disease",  # Placeholder
        "risk": risk,
        "score": score
    }
