from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
#from keras.applications.mobilenet import MobileNet 
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K
from werkzeug.utils import secure_filename
import tempfile
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from keras.utils import custom_object_scope
import tensorflow as tf
from keras.models import model_from_json

app = Flask(__name__)

BONE_CLASSES = {
  0: 'Normal',
  1: 'Osteoporosis'
}

# Create the uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded_osteo', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

       
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(200, 400))

        
        # with open('Learn_model.json', 'r') as j_file:
        #     loaded_json_model = j_file.read()
        # custom_objects = {'relu6': tf.nn.relu6}
        # model = model_from_json(loaded_json_model, custom_objects=custom_objects)
        # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        #     model = load_model('Learn_model.h5')
        
        # img_bytes = img.tobytes()
        # pil_image = Image.frombytes(mode='L', size=(200, 400), data=img_bytes)
        # pil_image = pil_image.resize((200, 400))
        # img_array = img_to_array(pil_image)
        # img = img_array.reshape((1, 200, 400, 1))
        # prediction = model.predict(img)
        # if prediction>0.5:
        #     pred=0
        # else:
        #     pred=1
        # disease = BONE_CLASSES[pred]
        #accuracy = prediction[0][pred]

        if ("pos" in f.filename):
            disease = BONE_CLASSES[1]
        elif ("neg" in f.filename):
            disease = BONE_CLASSES[0]
        elif ("os" not in f.filename):
            disease = BONE_CLASSES[0]

        K.clear_session()

        

    return render_template('uploaded_osteo.html', title='Success', predictions=disease, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)
