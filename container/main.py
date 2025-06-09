from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

from model import ImagePreprocessor, OnnxModel

# Initialize FastAPI
app = FastAPI()

# Load model and preprocessor
preprocessor = ImagePreprocessor(input_size=(224, 224))
model = OnnxModel("model.onnx")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    input_data = preprocessor.preprocess(image)
    
    # Predict
    output = model.predict(input_data)
    
    return {"output": output.tolist()}
