from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
from time import time
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
from flasgger import Swagger

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust CORS as needed for production
Swagger(app)  # Initialize Swagger UI

# Load the YOLOv5 model
model = YOLO('best.pt')

# Function to process uploaded image
def process_image(image_path):
    try:
        img = Image.open(image_path)
        results = model(img)  # Perform YOLOv5 inference
        detected_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detected_objects.append({
                    "class_name": class_name,
                    "confidence": confidence
                })

        return detected_objects

    except Exception as e:
        app.logger.error(f"Exception occurred during image processing: {str(e)}")
        return None

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload an image and get vegetable detection results
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The image file to upload
    responses:
      200:
        description: Detection results
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: object
                properties:
                  class_name:
                    type: string
                  confidence:
                    type: number
      400:
        description: No file part or no selected file
      404:
        description: No objects detected
      500:
        description: Internal server error
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file to a temporary location
        filename = secure_filename(file.filename)
        image_path = f"uploads/{int(time())}_{filename}"
        
        # Ensure the uploads directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        file.save(image_path)

        # Process the uploaded image
        detected_objects = process_image(image_path)

        # Remove the uploaded image after processing (optional)
        Path(image_path).unlink()

        if not detected_objects:
            return jsonify({"error": "No objects detected"}), 404
        else:
            return jsonify({"predictions": detected_objects}), 200

    except Exception as e:
        app.logger.error(f"Exception occurred during image processing: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def root():
    return (
        '<h2>Welcome to the Vegetable Detection API!</h2>'
        '<p>Visit the <a href="/apidocs">Swagger UI documentation</a> to try the API.</p>'
    )

if __name__ == '__main__':
    app.run(debug=True, port=os.environ.get('PORT', 5000))
