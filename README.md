# Image Classification Inference

## Setup Instructions

**Tested with:** Python 3.10

### Create a virtual environment

```bash
python -m venv ./venv
source ./venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Local inference testing
The generated ONNX model is available in the `container/` directory.
To test the ONNX model locally with a sample image:

```bash
python test.py --image <path-to-image>
```
## Remote inference testing
To test single image on Cerebrium endpoint

```bash
python test_server.py --url https://api.cortex.cerebrium.ai/v4/p-c193cdae/image-classification/infer --image <path-to-image>
```

### To run server-side test cases

```bash
python test_server.py --url https://api.cortex.cerebrium.ai/v4/p-c193cdae/image-classification/infer --run-tests
```
