"""
Unit tests for the ML model functions.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory to the path so we can import ml module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


@pytest.fixture
def sample_data():
    """
    Fixture that creates sample data for testing.
    """
    data = pd.DataFrame({
        'age': [39, 50, 38, 53, 28, 37, 49, 52],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 
                      'Private', 'Private', 'Private', 'Self-emp-not-inc'],
        'fnlgt': [77516, 83311, 215646, 234721, 338409, 284582, 160187, 209642],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 
                      'Masters', '9th', 'HS-grad'],
        'education-num': [13, 13, 9, 7, 13, 14, 5, 9],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 
                           'Married-civ-spouse', 'Married-civ-spouse', 
                           'Married-civ-spouse', 'Married-spouse-absent', 
                           'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 
                       'Handlers-cleaners', 'Prof-specialty', 'Exec-managerial', 
                       'Other-service', 'Exec-managerial'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 
                        'Wife', 'Wife', 'Own-child', 'Husband'],
        'race': ['White', 'White', 'White', 'Black', 'Black', 'White', 'Black', 'White'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male'],
        'capital-gain': [2174, 0, 0, 0, 0, 0, 0, 15024],
        'capital-loss': [0, 0, 0, 0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40, 40, 16, 45],
        'native-country': ['United-States', 'United-States', 'United-States', 
                          'United-States', 'Cuba', 'United-States', 'Jamaica', 
                          'United-States'],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '>50K']
    })
    return data


@pytest.fixture
def categorical_features():
    """
    Fixture for categorical features list.
    """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_train_model(sample_data, categorical_features):
    """
    Test that train_model returns a trained RandomForestClassifier model.
    """
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    model = train_model(X, y)
    
    # Check that the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    
    # Check that the model has been fitted
    assert hasattr(model, 'classes_')
    
    # Check that predictions can be made
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns correct metric values.
    """
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check that all metrics are 1.0 for perfect predictions
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
    
    # Check that metrics are between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_compute_model_metrics_imperfect():
    """
    Test compute_model_metrics with imperfect predictions.
    """
    # Some wrong predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check that metrics are between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    
    # Check that metrics are not perfect (should be less than 1)
    assert precision < 1.0 or recall < 1.0


def test_inference(sample_data, categorical_features):
    """
    Test that inference returns predictions of correct shape and type.
    """
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    model = train_model(X, y)
    predictions = inference(model, X)
    
    # Check that predictions is a numpy array
    assert isinstance(predictions, np.ndarray)
    
    # Check that predictions have the same length as input
    assert len(predictions) == len(X)
    
    # Check that predictions are binary (0 or 1)
    assert set(predictions).issubset({0, 1})


def test_process_data_training(sample_data, categorical_features):
    """
    Test process_data function in training mode.
    """
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    # Check that X is numpy array
    assert isinstance(X, np.ndarray)
    
    # Check that y is numpy array
    assert isinstance(y, np.ndarray)
    
    # Check that encoder is fitted
    assert hasattr(encoder, 'categories_')
    
    # Check that label binarizer is fitted
    assert hasattr(lb, 'classes_')
    
    # Check that the number of samples is correct
    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)


def test_process_data_inference(sample_data, categorical_features):
    """
    Test process_data function in inference mode.
    """
    # First process in training mode to get encoder and lb
    X_train, y_train, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    # Now test inference mode
    X_test, y_test, encoder_out, lb_out = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Check that returned encoder and lb are the same as input
    assert encoder_out is encoder
    assert lb_out is lb
    
    # Check shapes
    assert X_test.shape == X_train.shape
    assert y_test.shape == y_train.shape


def test_model_predictions_binary(sample_data, categorical_features):
    """
    Test that model predictions are binary values.
    """
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    model = train_model(X, y)
    predictions = inference(model, X)
    
    # Check all predictions are 0 or 1
    unique_predictions = np.unique(predictions)
    assert all(pred in [0, 1] for pred in unique_predictions)
