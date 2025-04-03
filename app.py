import cv2
import numpy as np
import easyocr
import joblib
import pandas as pd
import torch
from io import BytesIO
from PIL import Image
import requests
from flask import Flask, request, jsonify, make_response

# Initialize Flask app
app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/ocr', methods=['OPTIONS'])
def handle_options():
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Use GPU if available, but disable detector for lower memory consumption
use_gpu = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=use_gpu, detector=False)

def load_models():
    """Load models and data only when needed to reduce memory footprint."""
    try:
        vectorizer = joblib.load('vectorizer.pkl')
        knn = joblib.load('knn_model.pkl')
        df_iter = pd.read_csv('addresses.csv', iterator=True, chunksize=1000)
        df = pd.concat(df_iter)
        return vectorizer, knn, df
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

@app.route('/')
def index():
    return "Welcome to the OCR and Address Completion API!"

def fetch_image(image_url):
    """Download and process image efficiently."""
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return None
    except Exception:
        return None

@app.route('/ocr', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data['image_url']

    try:
        # Download the image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400
        
        # Convert to OpenCV format
        image = Image.open(BytesIO(response.content))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV format

        # Perform OCR
        results = reader.readtext(image)
        extracted_text = " ".join([text[1] for text in results])

        # Attempt address completion if text is detected
        completed_addresses = complete_address(extracted_text) if extracted_text else []

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "extracted_text": extracted_text,
        "completed_addresses": completed_addresses
    })


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
