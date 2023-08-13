from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize Flask app
app = Flask(__name__)

# Define endpoint for predicting image class
@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from request
    file = request.files['image']

    # Open the image file using PIL
    image = Image.open(file.stream).convert("RGB")

    # Preprocess the image
    size = (224, 224)
    resized_img = image.resize(size, resample=Image.LANCZOS)
    image = ImageOps.pad(resized_img, size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Return prediction as JSON
    response = {'class': class_name.strip()[2:], 'confidence': float(confidence_score)}
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
