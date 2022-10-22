from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from tqdm import tqdm_notebook as tqdm
import math 
import os
import sys 
# Some utilites
import numpy as np
import flask
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os

from keras.models import load_model 
from keras.preprocessing import image
import numpy as np



# Define a flask app
app = Flask(__name__)


image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder
# Model saved with Keras model.save()
MODEL_PATH = 'model/model.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
 #model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')




@app.route('/', methods=['GET','POST'])
def predict():
  # predicting images
  imagefile = request.files['imagefile']
  image_path = './static/images/' + imagefile.filename 
  imagefile.save(image_path)

  img = image.load_img(image_path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
  
  if classes[0]>0.5:
    return render_template('index.html', user_image=pic, prediction_text='{} is the image of Human'.format(imagefile.filename))
  else:
    return render_template('index.html', user_image=pic, prediction_text='{} is the image of Horse'.format(imagefile.filename))




if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()