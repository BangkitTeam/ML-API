from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import os
from datetime import datetime
import io

# Flask app initialization
app = Flask(__name__)

# Configure upload folder inside container
UPLOAD_FOLDER = '/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_H5_PATH = 'model_transferlearning.h5'
SAVED_MODEL_PATH = 'saved_model/my_model'

# Function to convert the .h5 file to SavedModel format if not already converted
def convert_h5_to_saved_model(h5_path, saved_model_path):
    if not os.path.exists(saved_model_path):
        try:
            print("Converting .h5 model to SavedModel format...")
            model = load_model(h5_path)
            model.save(saved_model_path)
            print(f"Model converted and saved at {saved_model_path}")
        except Exception as e:
            raise Exception(f"Error converting model: {str(e)}")
    else:
        print(f"SavedModel already exists at {saved_model_path}")

# Convert the model before starting the app
try:
    convert_h5_to_saved_model(MODEL_H5_PATH, SAVED_MODEL_PATH)
except Exception as e:
    print(f"Failed to convert the model: {str(e)}")
    exit(1)

# Load the converted SavedModel
model = tf.keras.models.load_model(SAVED_MODEL_PATH)

def decode_base64(base64_string):
    """
    Decode base64 string to binary data.
    """
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise Exception(f"Failed to decode base64: {str(e)}")

def generate_filename():
    """
    Generate a unique filename using timestamp.
    """
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

def save_image(image_data, filename):
    """
    Save image data to a file.
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        raise Exception(f"Failed to save image: {str(e)}")

def preprocess_image(file_path):
    """
    Load and preprocess image for prediction.
    """
    try:
        # Load the image and resize to the model's input size
        img = Image.open(file_path).resize((224, 224))  # Update size based on model input
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of a base64 image using the TensorFlow model.
    """
    try:
        # Get the JSON data
        data = request.get_json()

        # Validate request
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image found in request'}), 400

        # Process the base64 image
        base64_string = data['image']
        image_data = decode_base64(base64_string)
        filename = generate_filename()
        file_path = save_image(image_data, filename)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Predict using the TensorFlow model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1).tolist()
        confidence_scores = predictions.tolist()

        return jsonify({
            'status': 'ok',
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence_scores': confidence_scores
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
