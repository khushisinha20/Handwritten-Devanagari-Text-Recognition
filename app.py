from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load models
mnist_model = load_model('mnist_model.h5')
devanagari_model = load_model('devanagari_model.h5')

# Define label dictionaries
mnist_label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
devanagari_label_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'ण', 11: 'ब', 12: 'भ', 13: 'च', 14: 'छ', 15: 'क्ष', 16: 'द', 17: 'ड',
    18: 'ध', 19: 'ढ', 20: 'ग', 21: 'घ', 22: 'ज्ञ', 23: 'ह', 24: 'ज', 25: 'झ',
    26: 'क', 27: 'ख', 28: 'ङ', 29: 'ल', 30: 'म', 31: 'श', 32: 'न', 33: 'प',
    34: 'स', 35: 'ष', 36: 'फ', 37: 'र', 38: 'ट', 39: 'त',
    40: 'थ', 41: 'थ', 42: 'त्र', 43: 'व', 44: 'य', 45: 'ञ'
}


def preprocess(img, size):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((size, size))  # Resize image to model's input size
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array.reshape(1, size, size, 1)  # Reshape array to match model's input shape
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_mnist", methods=["POST"])
def predict_mnist():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    dataBytesIO.seek(0)
    image = Image.open(dataBytesIO)

    test_image = preprocess(image, 28)  # MNIST image size is 28x28

    prediction = mnist_model.predict(test_image)
    result = np.argmax(prediction, axis=1)[0]
    accuracy = float(np.max(prediction, axis=1)[0])

    label = mnist_label_dict[result]

    response = {'prediction': {'result': label, 'accuracy': accuracy}}

    return jsonify(response)

@app.route("/predict_devanagari", methods=["POST"])
def predict_devanagari():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    dataBytesIO = io.BytesIO(decoded)
    dataBytesIO.seek(0)
    image = Image.open(dataBytesIO)

    test_image = preprocess(image, 32) 

    prediction = devanagari_model.predict(test_image)
    result = np.argmax(prediction, axis=1)[0]
    accuracy = float(np.max(prediction, axis=1)[0])

    label = devanagari_label_dict[result]

    response = {'prediction': {'result': label, 'accuracy': accuracy}}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
