import base64
import numpy as np
import io
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import pandas as pd
import json
import csv
from tensorflow.keras import backend as K
from flask import request
from flask import jsonify
from flask import Flask, render_template

application = Flask(__name__)

# create dictionary with classes to later load the summary data from Redis database
summary_classes_dict = dict()

# create dictionary for mapping of network output index to class label
index_to_label_dict = dict()
FILEPATH_JSON_INDEX_CLASS = "index_to_label.json"

@application.route('/')
def my_form():
    return render_template('predict.html')

def get_model():
    global model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "model", "best_model"))
    print(" * Model loaded!")

def load_index_to_label_mapping():
    global index_to_label_dict
    with open(FILEPATH_JSON_INDEX_CLASS, "r") as json_file:
        index_to_label_dict = json.load(json_file)

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    # convert PIL image instance to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    # normalize input
    image /= 255.0
    # fit to expected input shape of model
    image = np.expand_dims(image, axis=0)

    return image

@application.route("/predict", methods=["POST"])
def predict():
    print("predict")
    message = request.get_json(force=True)
    encoded = message['image'].split(",")[1]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(256, 256))
    
    prediction = np.squeeze(model.predict(processed_image))

    winning_class_index = np.argmax(prediction)
    winning_class = index_to_label_dict[str(winning_class_index)]
    confidence = prediction[winning_class_index].astype(float)

    response = {
        'prediction': {
            'winning_class': winning_class,
            'confidence': confidence
        }
    }
    return jsonify(response)

def init_app():
    # load index to label mapping
    load_index_to_label_mapping()

    # load trained tensorflow model
    print(" * Loading Keras model...")
    get_model()

if __name__ == "__main__":
    init_app()

    # start flask app
    application.run(host='0.0.0.0')  