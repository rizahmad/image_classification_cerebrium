import torch
import onnx
import argparse
from image_classification_cerebrium.baseline.pytorch_model import Classifier, BasicBlock

model = Classifier(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load("./pytorch_model_weights.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ONNX model exported to model.onnx")

