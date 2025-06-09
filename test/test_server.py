import argparse
import requests
import time
from PIL import Image
import io
import os

CEREBRIUM_ENDPOINT = "0.0.0.0/infer"

def send_image(image_path):
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    files = {
        "file": (os.path.basename(image_path), image_bytes, "image/jpeg"),
    }

    start_time = time.time()

    response = requests.post(CEREBRIUM_ENDPOINT, files=files)

    elapsed_time = time.time() - start_time
    print(f"[INFO] Response time: {elapsed_time:.4f} seconds")

    if response.status_code != 200:
        print(f"[ERROR] Status code: {response.status_code}")
        print(f"[ERROR] Response content: {response.text}")
        return None

    result = response.json()
    print(f"[RESULT] Response JSON: {result}")

    output = result.get("output", [])
    if isinstance(output, list):
        class_id = int(max(enumerate(output), key=lambda x: x[1])[0])
        print(f"[PREDICTION] Class ID: {class_id}")
        return class_id
    else:
        print(f"[WARNING] Unexpected output format: {output}")
        return None

def run_custom_tests():
    print("\n=== Running preset tests ===\n")

    # Test 1: Health check (simple GET)
    print("[TEST 1] Health check...")
    try:
        health_response = requests.get(CEREBRIUM_ENDPOINT.replace("/infer", "/"))
        print(f"[INFO] Health check status code: {health_response.status_code}")
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")

    # Test 2: Latency test on known image
    print("\n[TEST 2] Latency test...")
    test_image_path = "n01440764_tench.jpeg"
    if not os.path.exists(test_image_path):
        print(f"[WARNING] Preset test image not found: {test_image_path}")
    else:
        for i in range(3):
            print(f"\n[Iteration {i+1}]")
            send_image(test_image_path)

    # (Optional) Test 3: Rate limit test (can simulate burst calls)
    print("\n[TEST 3] Burst test...")
    try:
        for i in range(5):
            print(f"[Burst call {i+1}]")
            send_image(test_image_path)
            time.sleep(0.5)  # small delay
    except Exception as e:
        print(f"[ERROR] Burst test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deployed Cerebrium model endpoint.")
    parser.add_argument("--url", type=str, help="Application URL.", required=True)
    parser.add_argument("--image", type=str, help="Path to image to send to model.", required=False)
    parser.add_argument("--run-tests", action="store_true", help="Run preset custom tests.")

    args = parser.parse_args()

    CEREBRIUM_ENDPOINT = args.url

    if args.image:
        print(f"[INFO] Sending image: {args.image}")
        send_image(args.image)

    if args.run_tests:
        run_custom_tests()
