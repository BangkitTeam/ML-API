import os
import pymysql
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Fetch configuration from environment variables (values set in docker-compose.yml)
db_host = os.getenv('DB_HOST', 'localhost')
db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', 'ikh123')
db_name = os.getenv('DB_NAME', 'coba1')

# Load the model once
model = tf.keras.models.load_model('garbage_classification_tf.h5')

# Class names for prediction
class_names = [
    "XLight", "bandaid", "battery", "bowlsanddishes", "bread", "bulb", "cans", "carton", 
    "chopsticks", "cigarettebutt", "diapers", "facialmask", "glassbottle", "leaflet", "leftovers", 
    "medicinebottle", "milkbox", "nailpolishbottle", "napkin", "newspaper", "nut", "penholder", 
    "pesticidebottle", "plasticbag", "plasticbottle", "plasticene", "rag", "tabletcapsule", 
    "thermometer", "toothbrush", "toothpastetube", "toothpick", "traditionalChinesemedicine", "watermelonrind"
]

# Database connection setup
def create_db_connection():
    try:
        return pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
        )
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e


# Fetch latest image URL from the database
def get_latest_image_url():
    db_connection = None  # Initialize db_connection to avoid reference error in finally block
    try:
        db_connection = create_db_connection()
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT id, image_url FROM upload ORDER BY createdAt DESC LIMIT 1;")
            result = cursor.fetchone()
            if result:
                return result  # Returning tuple (id, image_url)
            return None
    except Exception as e:
        print(f"Error fetching image URL: {e}")
        return None
    finally:
        if db_connection:
            db_connection.close()  # Only attempt to close if db_connection was initialized



# Download image from URL
def download_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Failed to download image, HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise e

# Preprocess image to fit model input
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model's expected input size
    image = np.array(image)

    if image.ndim == 2:  # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)

    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['GET'])
def predict_last_image():
    latest_image = get_latest_image_url()

    if latest_image:
        try:
            image_id, image_url = latest_image
            image = download_image_from_url(image_url)
            image = preprocess_image(image)

            predictions = model.predict(image)
            print(f"Predictions: {predictions}")  # Log raw predictions

            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            return jsonify({
                "id": image_id,
                "image_url": image_url,
                "prediction": predicted_label,
                "confidence": float(np.max(predictions))
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        print("No image found in database")
        return jsonify({"error": "No image found in database"}), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
