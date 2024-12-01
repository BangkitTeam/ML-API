import os
import pymysql
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

db_host = os.getenv('DB_HOST', 'localhost')  # For localhost
# db_host = os.getenv('DB_HOST', 'host.docker.internal')  # For Docker environment
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
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
        )
        logging.info("Database connection successful")
        return connection
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise e

# Insert prediction data into database
def insert_prediction(upload_id, user_id, prediction, confidence):
    try:
        connection = create_db_connection()
        with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO prediction (uploadId, userId, prediction, confidence)
                VALUES (%s, %s, %s, %s);
            """
            cursor.execute(insert_query, (upload_id, user_id, prediction, confidence))
            connection.commit()
            logging.info(f"Prediction for upload_id {upload_id} inserted successfully.")
    except Exception as e:
        logging.error(f"Error inserting prediction into database: {e}")
    finally:
        connection.close()


# Download image from URL
def download_image_from_url(url):
    try:
        logging.debug(f"Downloading image from URL: {url}") 
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            logging.debug(f"Image downloaded successfully from {url}")
            return image
        else:
            raise Exception(f"Failed to download image, HTTP Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise e


# Preprocess image to fit model input
def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image)

    if image.ndim == 2: 
        image = np.stack([image] * 3, axis=-1)

    image = image.astype('float32') / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.route('/latest-image', methods=['GET'])
def get_latest_image():
    db_connection = None
    try:
        db_connection = create_db_connection()
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT id, image_url FROM upload ORDER BY createdAt DESC LIMIT 1;")
            result = cursor.fetchone()
            if result:
                return jsonify({"id": result[0], "image_url": result[1]})
            else:
                return jsonify({"error": "No image found in database"}), 404
    except Exception as e:
        logging.error(f"Error fetching latest image from database: {e}")
        return jsonify({"error": "Error fetching image"}), 500
    finally:
        if db_connection:
            db_connection.close()

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

        logging.debug(f"Received image URL: {image_url}")

        image = download_image_from_url(image_url)
        image = preprocess_image(image)

        logging.debug(f"Image shape after preprocessing: {image.shape}")

        # Predict using the model
        predictions = model.predict(image)
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence = float(np.max(predictions))

        logging.debug(f"Predicted Class: {predicted_label} with confidence {confidence}")

        # Fetch the uploadId and userId (assuming this is linked with the image URL in the database)
        db_connection = create_db_connection()
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT id, userId FROM upload WHERE image_url = %s", (image_url,))
            result = cursor.fetchone()
            upload_id, user_id = result if result else (None, None)
        db_connection.close()

        if not upload_id:
            return jsonify({"error": "No matching upload found for the image URL"}), 404

        # Insert prediction data into the database
        insert_prediction(upload_id, user_id, predicted_label, confidence)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
