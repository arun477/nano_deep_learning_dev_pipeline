from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torchvision.transforms as transforms
import mnist_classifier

app = FastAPI()

app.mount("/static", StaticFiles(directory=Path("static")), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")

def process_image(file: UploadFile):
    image_bytes = file.file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    tensor_image = transform(pil_image)
    return tensor_image

@app.post("/predict")
async def predict(image: UploadFile):
    tensor_image = process_image(image)
    prediction = mnist_classifier.predict(tensor_image)
    return {"prediction": prediction}

