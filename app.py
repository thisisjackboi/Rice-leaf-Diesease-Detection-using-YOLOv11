from flask import Flask, request, jsonify, render_template
import os
from ultralytics import YOLO

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load rice leaf disease detection model
model = YOLO('best.pt')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Perform inference
    results = model(input_path)

    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy.cpu().numpy().tolist()[0]
            class_id = int(box.cls.cpu().numpy()[0])
            disease_type = model.names[class_id]
            detections.append({
                "disease_type": disease_type,
                "coordinates": coords
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(debug=True)
