from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict():
    return {"prediction": "Hello, World!"}
