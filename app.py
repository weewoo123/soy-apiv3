
import os
import numpy as np
from flask import Flask
from flask import request
from flask import json
from tensorflow import keras
from keras import models, metrics
from keras.metrics import Precision, Recall, SparseCategoricalAccuracy
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.utils import custom_object_scope, metrics_utils

#BUCKET_NAME = 'soybeanpredictor'
#MODEL_FILE_NAME = '404Soybeans.h5'
#MODEL_LOCAL_PATH = '/Users/mwhug/' + MODEL_FILE_NAME

app = Flask(__name__)

#creating custom metric for F1 Score, Precision, and Recall for analyzing model
class f1_score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(f1_score, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, label_true, label_pred, sample_weight=None):
        precisionf = self.precision(label_true, label_pred)
        recallf = self.recall(label_true, label_pred)
        self.f1.assign(2 * ((precisionf * recallf) / (precisionf + recallf + K.epsilon())))

    def result(self):
        return self.f1
    
    def reset_states(self):
        self.f1.assign(0.)
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        return {"f1": self.f1.numpy()}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class precision(keras.metrics.Metric):
    def __init__(self, name='precision'):
        super(precision, self).__init__(name=name)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.precision = self.add_weight(name='precision', initializer='zeros')

    def update_state(self, label_true, label_pred, sample_weight=None):
        tp = metrics_utils.ConfusionMatrix.TRUE_POSITIVES
        fp = metrics_utils.ConfusionMatrix.FALSE_POSITIVES
        self.true_positives.assign(tp)
        self.false_positives.assign(fp)
        self.precision.assign(tp / (tp + fp))

    def result(self):
        return self.precision
    
    def reset_states(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.precision.assign(0.)

    def get_config(self):
        return {"precision": self.precision.numpy()}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class recall(keras.metrics.Metric):
    def __init__(self, name='recall'):
        super(recall, self).__init__(name=name)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')

    def update_state(self, label_true, label_pred, sample_weight=None):
        tp = metrics_utils.ConfusionMatrix.TRUE_POSITIVES
        fn = metrics_utils.ConfusionMatrix.FALSE_NEGATIVES
        self.true_positives.assign(tp)
        self.false_negatives.assign(fn)
        self.recall.assign(tp / (tp + fn))

    def result(self):
        return self.recall
    
    def reset_states(self):
        self.true_positives.assign(0.)
        self.false_negatives.assign(0.)
        self.recall.assign(0.)

    def get_config(self):
        return {"recall": self.recall.numpy()}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = keras.models.load_model("404Soybeans.h5", custom_objects={"precision": precision, "recall": recall})
        
def model_predict(img_path):
    print("Getting Prediction!")
    #load the image, make sure it is the target size (specified by model code)
    img = keras.utils.load_img(img_path, target_size=(256,256))
    #convert the image to an array
    img = keras.utils.img_to_array(img)
    #normalize array size
    img /= 255           
    #expand image dimensions for keras convention
    img = np.expand_dims(img, axis = 0)

    #call model for prediction
    #opt = keras.optimizers.RMSprop(learning_rate = 0.01)
    with custom_object_scope({"precision": precision, "recall": recall}):
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [SparseCategoricalAccuracy(), precision(), recall()])
    #model = load_model()
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
        msg = 'Model Prediction: Your plant is within Day 9 and Day 11 of the growth cycle.'
    elif index == 1:
        #output this range
        msg =  'Model Prediction: Your plant is within Day 12 and Day 14 of the growth cycle.'
    elif index == 2:
        #output this range
        msg =  'Model Prediction: Your plant is within Day 15 and Day 17 of the growth cycle.'
    elif index == 3:
        #output this range
        msg =  'Model Prediction: Your plant is within Day 18 and Day 20 of the growth cycle.'
    elif index == 4:
        #output this range
        msg =  'Model Prediction: Your plant is within Day 22 and Day 28 of the growth cycle.'
    else:
        msg =  'Error: Model sent prediction out of the prescribed range. Please try again.'

    return {"message": msg, "accuracy": compareVal}

#def load_model():
        #print('Loading model from S3')
        # connection = S3Connection()
        # s3bucket = connection.create_bucket(BUCKET_NAME)
        # keyobject = Key(s3bucket)
        # keyobject.key = MODEL_FILE_NAME

        # contents = keyobject.get_contents_to_filename(MODEL_LOCAL_PATH)
        # model = joblib.load(MODEL_LOCAL_PATH)
        # return model

        

#def predict(img):

#         print('Making predictions')
#         #data coming in as json. Need to convert to numpy array (normal array might work, not sure which is most ideal yet)
#         #model_input = np.asarray(input)

#         pred = model_predict(model_input)
#         output = output_statement(pred)
#         return output

@app.route('/', methods=['POST', 'GET'])
def index():
        output = {}
        if request.method == 'POST':
            img = request.files['image']
            print(img)

            basepath = os.path.dirname(__file__)
            img_path = os.path.join(basepath, 'uploads', secure_filename(img.filename))
            print(img_path)
            img.save(img_path)
            pred = model_predict(img_path)
            print("Received a prediction!")
            pred = pred.tolist()
            print("Prediction received from model (raw): ", pred)
            output = output_statement(pred)
            os.remove(img_path)
            output = {"message": output["message"], "accuracy": output["accuracy"]}
            print(output)
            return output
            # payload = json.loads(img)
            # input = payload['payload']  #variable input is a dictionary.
            # prediction = predict(input)
            # data = {}
            # data['data'] = prediction[-1]
            #return "json.dumps(data)"

        elif request.method == 'GET':
            response = output
            response["MESSAGE"] = "Soybean Prediction API is running!"
            return response