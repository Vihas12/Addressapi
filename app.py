import os
import gc
from flask import Flask, request, jsonify
import easyocr
import joblib
import pandas as pd
import torch

# Initialize Flask app
app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

class OCRModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if not self._initialized:
            # Initialize only what's needed
            self.use_gpu = torch.cuda.is_available()
            self.reader = None  # Will be loaded on first use
            self.vectorizer = None
            self.knn = None
            self.df = None
            self._initialized = True
    
    def get_reader(self):
        if self.reader is None:
            # Load EasyOCR only when needed
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        return self.reader
    
    def load_models(self):
        if self.vectorizer is None:
            try:
                # Load models with memory mapping for large files
                self.vectorizer = joblib.load('./vectorizer.pkl', mmap_mode='r')
                self.knn = joblib.load('./knn_model.pkl', mmap_mode='r')
                
                # Replace your CSV loading code with:
                self.df = pd.read_parquet('./addresses.parquet')
                
                # Reduce memory usage by optimizing data types
                for col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        self.df[col] = self.df[col].astype('category')
                
            except Exception as e:
                print(f"Error loading models: {e}")
                self.vectorizer, self.knn, self.df = None, None, None
                raise

    def complete_address(self, query):
        self.load_models()
        if self.vectorizer is None or self.knn is None or self.df is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        _, idx = self.knn.kneighbors(query_vector)
        results = self.df.iloc[idx[0]]["full_address"].tolist()
        
        # Explicit cleanup
        del query_vector, idx
        gc.collect()
        
        return results

# Initialize the model manager
model_manager = OCRModelManager()
model_manager.initialize()

@app.route('/')
def health_check():
    return "status: ok"

@app.route('/ocr', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    try:
        # Get OCR reader (loaded on first use)
        reader = model_manager.get_reader()
        results = reader.readtext(data['image_url'])
        extracted_text = " ".join([text[1] for text in results])

        # Only complete address if text was found
        completed_addresses = []
        if extracted_text.strip():
            completed_addresses = model_manager.complete_address(extracted_text)
        
        # Clean up memory
        del results
        gc.collect()

        return jsonify({
            "extracted_text": extracted_text,
            "completed_addresses": completed_addresses
        })

    except Exception as e:
        # Ensure models are cleared if error occurs
        model_manager.reader = None
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use gevent for better memory management with multiple requests
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('0.0.0.0', 7860), app)
    http_server.serve_forever()
