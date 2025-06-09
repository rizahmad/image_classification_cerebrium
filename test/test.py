import argparse
import time
import os
from PIL import Image
import numpy as np

from model import ImagePreprocessor, OnnxModel

MODEL_PATH = "model.onnx"
INPUT_SIZE = (224, 224)

def run_single_test(image_path):
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    assert os.path.exists(MODEL_PATH), f"ONNX model not found: {MODEL_PATH}"

    # Step 1: Initialize components
    preprocessor = ImagePreprocessor(input_size=INPUT_SIZE)
    model = OnnxModel(MODEL_PATH)

    # Step 2: Load and preprocess image
    print(f"[INFO] Loading image: {image_path}")
    image = Image.open(image_path)
    input_data = preprocessor.preprocess(image)
    
    print(f"[DEBUG] Preprocessed input shape: {input_data.shape}")
    assert input_data.shape[0] == 1, "Batch size must be 1"

    # Step 3: Run inference
    print("[INFO] Running model inference...")
    start_time = time.time()
    output = model.predict(input_data)
    elapsed_time = time.time() - start_time

    # Step 4: Print results
    print(f"[INFO] Inference completed in {elapsed_time:.4f} seconds")
    print(f"[RESULT] Output shape: {output.shape}")

    # Step 5: Basic sanity checks
    assert isinstance(output, np.ndarray), "Output must be a numpy array"
    assert output.ndim in [1, 2], "Output must be 1D (classification) or 2D (batch_size, classes)"
    assert not np.isnan(output).any(), "Output contains NaNs!"

    # Step 6: Get predicted class
    if output.ndim == 2:
        # Assume (batch_size, num_classes)
        class_id = int(np.argmax(output[0]))
    else:
        # Assume (num_classes,)
        class_id = int(np.argmax(output))

    print(f"[PREDICTION] Class ID: {class_id}")
    return class_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX model locally.")
    parser.add_argument("--image", type=str, help="Path to image to test.", required=True)

    args = parser.parse_args()

    run_single_test(args.image)
