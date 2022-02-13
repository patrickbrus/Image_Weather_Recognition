import base64
import numpy as np
import io
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import pandas as pd
import redis
import json
import csv
from tensorflow.keras import backend as K
from flask import request
from flask import jsonify
from flask import Flask, render_template

application = Flask(__name__)

# create redis instance
redis_inst = redis.Redis(host="redis-cluster-ip-service")

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

def init_classes_dict():
    global summary_classes_dict
    for value in index_to_label_dict.values():
        if value not in summary_classes_dict.keys():
            summary_classes_dict[value] = 0

def setup_redis_db():
    # loop over summary_classes_dict and add key-value pair if not already exists
    for key, value in summary_classes_dict.items():
        if redis_inst.get(key) == None:
            redis_inst.set(key, value)

def load_redis_db_vals():
    results_dict = {}
    for key in summary_classes_dict.keys():
        results_dict[key] = int(redis_inst.get(key))
    
    return results_dict

def update_redis_db(key):
    # get current value for key
    current_val = int(redis_inst.get(key))
    
    # increment current value
    current_val += 1

    # write to redis database again
    redis_inst.set(key, str(current_val))

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

@application.route("/load", methods=["POST"])
def send_summary():
    print("load and send past prediction summary")
    response = load_redis_db_vals()
    return jsonify(response)


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

    # update redis database
    update_redis_db(winning_class)

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

    # initialize summary_dict
    init_classes_dict()

    # setup redis database
    setup_redis_db()

    # load trained tensorflow model
    print(" * Loading Keras model...")
    get_model()

if __name__ == "__main__":
    init_app()

    # start flask app
    application.run(host='0.0.0.0')  