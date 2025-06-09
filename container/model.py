import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

class ImagePreprocessor:
    def __init__(self, input_size=(224, 224)):
        """
        Preprocesses the input for the model.
        """
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesses a PIL image and returns a NumPy array suitable for ONNX model input.
        """

        tensor = self.transform(image).unsqueeze(0)
        return tensor.numpy()


class OnnxModel:
    def __init__(self, model_path: str):
        """
        Loads an ONNX model for inference.
        """
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs prediction using the ONNX model.
        """
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]



