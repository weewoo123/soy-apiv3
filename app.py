from flask import Flask
import tensorflow as tf
import numpy as np
import keras
model = keras.models.load_model("soybeans.h5")

app = Flask(__name__)

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
def predict():
    return "Hello World!"