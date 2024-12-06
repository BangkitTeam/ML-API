import os
import logging
import tensorflow as tf
from flask import Flask, jsonify, request
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import tempfile

# Memuat variabel dari file .env
load_dotenv()

# Ambil URL atau path model dari variabel lingkungan
model_url = os.getenv('MODEL_PATH')

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Class names for prediction
class_names = [
    'bottlecap', 'cans', 'cardboard', 'ceramicsbowl', 'disc', 
    'galvanizedsteel', 'glassbottle', 'newspaper', 'paper', 
    'pen', 'plasticbag', 'plasticbottle', 'rag', 'spoonfork', 
    'tire', 'watergallon'
]

# Function to download model from URL and load it into TensorFlow
def download_model_from_url(model_url):
    try:
        logging.debug(f"Downloading model from URL: {model_url}")
        response = requests.get(model_url)
        if response.status_code == 200:
            # Simpan model yang diunduh ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_model_file:
                temp_model_file.write(response.content)
                temp_model_path = temp_model_file.name
                logging.debug(f"Model successfully downloaded and saved to {temp_model_path}")
            
            # Muat model menggunakan path file sementara
            model = tf.keras.models.load_model(temp_model_path)
            logging.debug("Model successfully loaded.")
            return model
        else:
            raise Exception(f"Failed to download model, HTTP Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        raise e

# Muat model menggunakan URL yang diambil dari .env
model = download_model_from_url(model_url)

# Function to download image from URL
def download_image_from_url(url):
    try:
        logging.debug(f"Downloading image from URL: {url}") 
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            logging.debug(f"Image successfully downloaded from {url}")
            return image
        else:
            raise Exception(f"Failed to download image, HTTP Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise e

# Function to preprocess image to match model input
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model
    image = np.array(image)

    if image.ndim == 2:  # If image is grayscale, convert to RGB
        image = np.stack([image] * 3, axis=-1)

    image = image.astype('float32') / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        
        if not data:
            return jsonify({"error": "Request body must be in JSON format"}), 400

        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        logging.debug(f"Image URL received: {image_url}")

        image = download_image_from_url(image_url)
        image = preprocess_image(image)

        logging.debug(f"Image shape after preprocessing: {image.shape}")

        # Perform prediction using the model
        predictions = model.predict(image)
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence = int(np.max(predictions) * 100)  # Convert to percentage without decimals

        logging.debug(f"Predicted Class: {predicted_label} with confidence {confidence}")

        # Return prediction response
        return jsonify({
            "image_url": image_url,
            "prediction": predicted_label,
            "confidence": confidence
        })

    except ValueError as ve:
        logging.error(f"Error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

# Run Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
