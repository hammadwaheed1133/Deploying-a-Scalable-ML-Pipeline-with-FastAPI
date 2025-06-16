# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InferenceInput(BaseModel):
    age: int
    education: str
    occupation: str

@app.get("/")


def root():
    return {"message": "Hello from the API!"}

@app.post("/predict")


def predict(input_data: InferenceInput):
    # Dummy logic for inference
    if input_data.age > 40:
        return {"result": ">50K"}
    else:
        return {"result": "<=50K"}