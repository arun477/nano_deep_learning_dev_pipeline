from fastapi import FastAPI, UploadFile
from PIL import Image
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import mnist_classifier
import torch
from pathlib import Path
import datetime
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory=Path("static")), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")

upload_dir = Path("uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

def process_image(file):
    image = Image.open(file.file)
    image = image.resize((28, 28))  # Resize to MNIST image size
    image = image.convert("L")  # Convert to grayscale
    raw_image = image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    return torch.from_numpy(image).float().reshape(1, 28, 28), raw_image
    
def store_img(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}.png"
    output_path = upload_dir / unique_filename
    image.save(output_path)
    
@app.post("/predict")
async def predict(image:UploadFile):
    tensor_image, raw_image = process_image(image)
    prediction = mnist_classifier.predict(tensor_image)
    store_img(raw_image)
    return {"prediction": prediction}