from flask import Flask,render_template,request,jsonify
import pickle
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import json

img_size=28

app = Flask(__name__) 

model = load_model('mnist_model.h5')
label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


def preprocess(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((img_size, img_size))  # Resize image to model's input size
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array.reshape(1, img_size, img_size, 1)  # Reshape array to match model's input shape
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array
	

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=True)
