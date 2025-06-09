from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
from pydantic import BaseModel

from model import ImagePreprocessor, OnnxModel

# Initialize FastAPI
app = FastAPI()

# Load model and preprocessor
preprocessor = ImagePreprocessor(input_size=(224, 224))
model = OnnxModel("model.onnx")

#@app.get("/")
def health_check():
    return {"status": "ok"}

class InferenceRequest(BaseModel):
    image_base64: str

#@app.post("/infer")
async def infer(request):
    # Read JSON payload
    payload = await request.json()
    image_base64 = payload.get("image_base64")
    if image_base64 is None:
            return {"error": "Missing 'image_base64' field"}
    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)

    # Read image file
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    input_data = preprocessor.preprocess(image)
    
    # Predict
    output = model.predict(input_data)
    
    return {"output": output.tolist()}
