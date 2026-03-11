"""
Unit tests for the FastAPI application.
"""

from urllib import response

import pytest
from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)


def test_get_root():
    """
    Test GET method on the root endpoint.
    Tests both status code and response content.
    """
    response = client.get("/")
    
    # Test status code
    assert response.status_code == 200
    
    # Test response body
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]


def test_post_predict_low_income():
    """
    Test POST method for prediction of income <=50K.
    Uses data that should predict income below 50K.
    """
    # Sample input for low income prediction
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)
    
    # Test status code
    assert response.status_code == 200
    
    # Test response structure
    assert "prediction" in response.json()
    
    # test_post_predict_low_income
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_high_income():
    """
    Test POST method for prediction of income >50K.
    Uses data that should predict income above 50K.
    """
    # Sample input for high income prediction
    # Professional with high education and capital gains
    data = {
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
    
    response = client.post("/predict", json=data)
    
    # Test status code
    assert response.status_code == 200
    
    # Test response structure
    assert "prediction" in response.json()
    
    # test_post_predict_high_income
    assert response.json()["prediction"] == ">50K"


def test_post_predict_with_hyphens():
    """
    Test POST method handles field names with hyphens correctly.
    """
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
    
    response = client.post("/predict", json=data)
    
    # Test that request with hyphenated field names works
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_get_root_content():
    """
    Additional test for GET method to check specific content.
    """
    response = client.get("/")
    
    json_response = response.json()
    
    # Check both status code and body content
    assert response.status_code == 200
    assert "message" in json_response
    assert "description" in json_response


def test_post_predict_validation():
    """
    Test POST method with example from schema.
    """
    # Using the exact example from the Pydantic model
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
    
    response = client.post("/predict", json=data)
    
    # Validate response
    assert response.status_code == 200
    assert "prediction" in response.json()
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]
