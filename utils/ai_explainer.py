# utils/ai_explainer.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# API headers for authentication and identification
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",  # Authentication
    "HTTP-Referer": "https://yourproject.com",        # Required by OpenRouter
    "X-Title": "Disease Risk Predictor"               # Project identification
}

# List of available free/cheaper models in order of preference
AVAILABLE_MODELS = [
    "mistralai/mistral-7b-instruct",
    "openchat/openchat-7b",
    "gryphe/mythomist-7b",
    "nousresearch/nous-hermes-llama2-13b"
]

# Main function to get AI explanations
def get_ai_explanation(prompt, model=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Try each model in sequence if no specific model is requested
    models_to_try = [model] if model else AVAILABLE_MODELS
    
    last_error = None
    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            continue
    
    return f"AI explanation error: All models failed. Last error: {last_error}"

