import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

try:
    response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
    print(response.json())
except Exception as e:
    print(f"Error checking models: {e}")
