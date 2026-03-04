# Put the code for your API here.
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Determine if running on Heroku or locally
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load model artifacts
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "model", "encoder.pkl")
lb_path = os.path.join(os.path.dirname(__file__), "model", "lb.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

with open(lb_path, "rb") as f:
    lb = pickle.load(f)

# Import from local modules
from starter.ml.data import process_data
from starter.ml.model import inference

# Create FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting whether income exceeds $50K/year based on census data",
    version="1.0.0"
)


# Pydantic model for input data with Field aliases to handle hyphens
class CensusInput(BaseModel):
    age: int = Field(..., example=37)
    workclass: Literal[
        "State-gov", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "Private", "Without-pay", "Never-worked"
    ] = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: Literal[
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ] = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num", example=9)
    marital_status: Literal[
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse"
    ] = Field(..., alias="marital-status", example="Never-married")
    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ] = Field(..., example="Adm-clerical")
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ] = Field(..., example="Not-in-family")
    race: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
        "Black", "Other"
    ] = Field(..., example="White")
    sex: Literal["Female", "Male"] = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: Literal[
        "United-States", "Cambodia", "England", "Puerto-Rico",
        "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
        "Japan", "Greece", "South", "China", "Cuba", "Iran",
        "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
        "Vietnam", "Mexico", "Portugal", "Ireland", "France",
        "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
        "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
        "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ] = Field(..., alias="native-country", example="United-States")

    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # JSON schema for documentation
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    prediction: str = Field(..., example="<=50K")


@app.get("/", response_model=dict, status_code=200)
async def root() -> dict:
    """
    Welcome message for the API root endpoint.
    
    Returns:
        dict: Welcome message
    """
    return {
        "message": "Welcome to the Census Income Prediction API!",
        "description": "Use POST /predict to make predictions about income levels."
    }


@app.post("/predict", response_model=PredictionOutput, status_code=200)
async def predict(input_data: CensusInput) -> PredictionOutput:
    """
    Predict whether income exceeds $50K/year based on census data.
    
    Args:
        input_data: Census data input following the CensusInput model
        
    Returns:
        PredictionOutput: Prediction result (<=50K or >50K)
    """
    # Convert input to DataFrame
    input_dict = {
        "age": [input_data.age],
        "workclass": [input_data.workclass],
        "fnlgt": [input_data.fnlgt],
        "education": [input_data.education],
        "education-num": [input_data.education_num],
        "marital-status": [input_data.marital_status],
        "occupation": [input_data.occupation],
        "relationship": [input_data.relationship],
        "race": [input_data.race],
        "sex": [input_data.sex],
        "capital-gain": [input_data.capital_gain],
        "capital-loss": [input_data.capital_loss],
        "hours-per-week": [input_data.hours_per_week],
        "native-country": [input_data.native_country]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # Process the input data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    prediction = inference(model, X)
    
    # Convert prediction to label
    prediction_label = lb.inverse_transform(prediction)[0]
    
    return PredictionOutput(prediction=prediction_label)

