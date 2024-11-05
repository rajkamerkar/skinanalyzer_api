import os
import cv2
import requests
from ultralytics import YOLO
from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Initialize the YOLO model
model_path = 'yolov11x1.1-trained.pt'  # Ensure this model file is in the same directory
model = YOLO(model_path)

# Temporary fix: add placeholder names for missing classes
expected_classes = 13  # Set this to the correct number of classes
for i in range(expected_classes):
    if i not in model.names:
        model.names[i] = f"class_{i}"

def annotate_image(img, confidence=0.25):
    """Runs YOLO model to detect skin issues with specified confidence and returns the annotated image."""
    # Run YOLO model inference
    results = model.predict(img, conf=confidence)
    # Get the annotated image from results
    annotated_img = results[0].plot()
    return annotated_img

def save_to_github(image_data, image_name):
    """Uploads the image to a GitHub repository and returns the image URL."""
    # Set up GitHub repository details
    github_token = "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"  # Replace with your GitHub token
    repo_owner = "rajkamerkar"
    repo_name = "skinanalyzer_api"
    branch_name = "main"

    # Prepare the GitHub API URL
    url = f"https://api.github.com/repos/rajkamerkar/skinanalyzer_api/contents/results/{image_name}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Convert the image to bytes
    image_bytes = cv2.imencode('.jpeg', image_data)[1].tobytes()
    content = BytesIO(image_bytes).read()
    # Create data payload for the API request
    payload = {
        "message": f"Adding {image_name}",
        "content": content.encode("base64"),
        "branch": branch_name
    }
    
    # Send a PUT request to GitHub API to create/update the file
    response = requests.put(url, headers=headers, json=payload)
    
    if response.status_code == 201 or response.status_code == 200:
        print(f"Uploaded {image_name} to GitHub.")
        # Return the URL to the uploaded file
        return f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch_name}/results/{image_name}"
    else:
        raise Exception("Failed to upload image to GitHub", response.json())

@app.route('/annotate-image', methods=['POST'])
def handle_image_annotation():
    """Handles image upload, annotation, and GitHub upload, returning URL of annotated image."""
    try:
        # Get the uploaded image and confidence threshold
        file = request.files['image']
        confidence = float(request.form.get('confidence', 0.25))

        # Read image into OpenCV format
        img = Image.open(file.stream)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Annotate the image
        annotated_img = annotate_image(img, confidence)

        # Generate a unique filename for GitHub
        image_name = f"annotated_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpeg"
        
        # Save annotated image to GitHub and get the image URL
        image_url = save_to_github(annotated_img, image_name)

        # Return the image URL as JSON response
        return jsonify({"status": "success", "image_url": image_url})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
