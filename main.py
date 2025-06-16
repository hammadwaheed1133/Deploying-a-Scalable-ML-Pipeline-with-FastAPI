from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class InferenceInput(BaseModel):
    age: int
    education: str
    occupation: str


@app.get("/")
def root():
    return {"message": "Hello"}


@app.post("/predict")
def predict(input_data: InferenceInput):
    return {"result": ">50K"}
