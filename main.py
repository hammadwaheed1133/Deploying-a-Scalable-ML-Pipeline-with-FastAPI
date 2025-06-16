from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml.model import load_model, inference
from ml.data import process_data



app = FastAPI()

# Define the expected request body using Pydantic
class InferenceRequest(BaseModel):
    age: int
    education: str
    occupation: str

# GET request on the root giving a welcome message
@app.get("/")
def read_root():
    return {"message": "Hello from the API!"}

# POST request that makes model inference
@app.post("/predict")
def predict(input: InferenceRequest):
    # Dummy logic: age > 40 means income > 50K
    if input.age > 40:
        return {"result": ">50K"}
    else:
        return {"result": "<=50K"}
