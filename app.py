from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
from time import time
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust CORS as needed for production

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

if __name__ == '__main__':
    app.run(debug=True, port=os.environ.get('PORT', 5000))
