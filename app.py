from flask import Flask
import tensorflow as tf
import numpy as np
import keras
import sys
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

model = keras.models.load_model("soybeans.h5")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def model_predict(img_path):
    #load the image, make sure it is the target size (specified by model code)
    img = keras.utils.load_img(img_path, target_size=(224,224))
    #convert the image to an array
    img = keras.utils.img_to_array(img)
    #normalize array size
    img /= 255           
    #expand image dimensions for keras convention
    img = np.expand_dims(img, axis = 0)

    #call model for prediction
    opt = keras.optimizers.RMSprop(learning_rate = 0.01)

    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    sys.stdout.write("predicting")
    sys.stdout.flush()
    pred = model.predict(img)

    return pred

def output_statement(pred):
    index = -1
    compareVal = -1
    for i in range(len(pred[0])):
        if(compareVal < pred[0][i]):
            compareVal = pred[0][i]
            index = i
    if index == 0:
        #output this range of days
        msg = '9-12'
    elif index == 1:
        #output this range
        msg = '13-16'
    elif index == 2:
        #output this range
        msg = '17-20'
    elif index == 3:
        #output this range
        msg = '21-28'
    else:
        return '[ERROR]: Out of range'
    return {"prediction": msg, "accuracy": compareVal}
#
@app.route("/")
def root():
    return "API is up and running!"

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        file = request.files['image']
        sys.stdout.write("hi")
        sys.stdout.flush()
        #need to get image from POST request
        # #create img_path to call model
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))            
        file.save(img_path)
        # #call model
        pred = model_predict(img_path)
        pred = pred.tolist()

        values = output_statement(pred)
        os.remove(img_path)
        return {"id":1, "filename": file.filename, "prediction": values["prediction"], "accuracy": values["accuracy"]}
    if request.method == "GET":
        return "Predictions are up and running"