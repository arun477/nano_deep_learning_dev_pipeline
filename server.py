from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

app.mount("/static", StaticFiles(directory=Path("static")), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict():
    return {"prediction": "Hello, World!"}
