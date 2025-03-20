from flask import Flask, request, jsonify, render_template
import os
from ultralytics import YOLO
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (for API access)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

# Allowed file types for security
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Only PNG, JPG, and JPEG are allowed."}), 400

        # Save the uploaded file
        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Perform inference
        results = model(input_path)

        # Extract detections
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

        # Return JSON output
        response = {"detections": detections}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
