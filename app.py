import cv2
import numpy as np
import easyocr
import joblib
import pandas as pd
import torch
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
    image = fetch_image(image_url)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    try:
        results = reader.readtext(image)
        extracted_text = " ".join([text[1] for text in results]) if results else ""

        completed_addresses = []
        if extracted_text:
            vectorizer, knn, df = load_models()
            if vectorizer and knn and df is not None:
                query_vector = vectorizer.transform([extracted_text])
                _, idx = knn.kneighbors(query_vector)
                completed_addresses = df.iloc[idx[0]]["full_address"].tolist()
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "extracted_text": extracted_text,
        "completed_addresses": completed_addresses
    })

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
