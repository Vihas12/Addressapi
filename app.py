import requests
import os
import cv2
import numpy as np
import easyocr
import joblib
import pandas as pd
import torch
from flask import Flask, request, jsonify, make_response

# Initialize Flask app
app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Use GPU if available
use_gpu = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=use_gpu)

vectorizer, knn, df = None, None, None  # Prevent unnecessary memory usage

def load_models():
    """Load models only when required to save memory."""
    global vectorizer, knn, df
    if vectorizer is None or knn is None or df is None:
        try:
            vectorizer = joblib.load('./uploads/vectorizer.pkl')
            knn = joblib.load('./uploads/knn_model.pkl')
            df = pd.read_parquet('./uploads/addresses.parquet')

        except Exception as e:
            print(f"Error loading models: {e}")
            vectorizer, knn, df = None, None, None

def complete_address(query):
    load_models()  # Load models only when needed
    if vectorizer is None or knn is None or df is None:
        return []
    
    query_vector = vectorizer.transform([query])
    _, idx = knn.kneighbors(query_vector)
    results = df.iloc[idx[0]]["full_address"].tolist()

    # Free memory after processing
    del query_vector, idx
    return results


@app.route('/')
def health_check():
    return "status: ok"

@app.route('/ocr', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url = data['image_url']
    
    try:
        results = reader.readtext(image_url)
        extracted_text = " ".join([text[1] for text in results])

        # Attempt address completion if text is detected
        completed_addresses = complete_address(extracted_text) if extracted_text else []

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "extracted_text": extracted_text,
        "completed_addresses": completed_addresses
    })

# Run the Flask app in production
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

