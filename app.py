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
    10: 'adna', 11: 'ba', 12: 'bha', 13: 'cha', 14: 'chha', 15: 'chhya', 16: 'da', 17: 'daa',
    18: 'dha', 19: 'dhaa', 20: 'ga', 21: 'gha', 22: 'gya', 23: 'ha', 24: 'ja', 25: 'jha',
    26: 'k', 27: 'kha', 28: 'kna', 29: 'la', 30: 'ma', 31: 'motosaw', 32: 'na', 33: 'pa',
    34: 'patalosaw', 35: 'petchiryakha', 36: 'pha', 37: 'ra', 38: 'taamatar', 39: 'tabala',
    40: 'tha', 41: 'thaa', 42: 'tra', 43: 'waw', 44: 'yaw', 45: 'yna'
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
