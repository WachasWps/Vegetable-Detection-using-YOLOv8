from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Load the trained model
model = YOLO('best.pt')

# Define the path to the data.yaml file
data_path = 'C:/Users/wacha/OneDrive/Desktop/Storage/Kleos/custom/datasets/data.yaml'

# Define a function to process an uploaded image
def process_image(image_path):
    results = model.predict(source=image_path, conf=0.25, save=True, save_txt=True)

    detected_objects = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy array if necessary
        confidences = result.boxes.conf.cpu().numpy()  # Convert to numpy array if necessary
        classes = result.boxes.cls.cpu().numpy()  # Convert to numpy array if necessary

        for i in range(len(boxes)):
            class_id = int(classes[i])
            confidence = float(confidences[i])
            class_name = model.names[class_id]

            detected_objects.append({
                "class_name": class_name,
                "confidence": confidence
            })

    return detected_objects

# Route to upload an image and get predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        image_path = f"uploads/{filename}"  # Save uploaded file to a directory
        file.save(image_path)

        # Process the uploaded image
        detected_objects = process_image(image_path)

        # Return JSON response
        return jsonify({"predictions": detected_objects}), 200

    return jsonify({"error": "Unknown error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
