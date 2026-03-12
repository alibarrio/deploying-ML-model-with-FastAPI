"""
Script to test the live API deployed on a cloud platform.

This script sends a POST request to the live API to test model inference.
Update the API_URL variable with your deployed API URL.
"""

import requests
import json

# Update this URL with your deployed API URL
# For Heroku: https://your-app-name.herokuapp.com
# For Render: https://your-app-name.onrender.com
API_URL = "https://census-income-api-9lip.onrender.com"

# Test data for prediction
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

print("Testing live API...")
print(f"API URL: {API_URL}")
print(f"\nSending POST request to {API_URL}/predict")
print(f"Input data:\n{json.dumps(data, indent=2)}")

try:
    # Send POST request
    response = requests.post(f"{API_URL}/predict", json=data)
    
    # Print response
    print(f"\nResponse Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ Request successful!")
        print(f"Prediction: {response.json()['prediction']}")
    else:
        print("✗ Request failed!")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n✗ Error: Could not connect to the API.")
    print("Make sure the API_URL is correct and the server is running.")
except Exception as e:
    print(f"\n✗ Error: {str(e)}")

print("\n" + "="*50)

# Test another example - high income profile
print("\nTesting high income profile...")
high_income_data = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

print(f"Input data:\n{json.dumps(high_income_data, indent=2)}")

try:
    response = requests.post(f"{API_URL}/predict", json=high_income_data)
    print(f"\nResponse Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ Request successful!")
        print(f"Prediction: {response.json()['prediction']}")
    else:
        print("✗ Request failed!")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n✗ Error: Could not connect to the API.")
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
