from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
from keras.applications.mobilenet import MobileNet 
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K
from werkzeug.utils import secure_filename
import tempfile
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

BONE_CLASSES = {
  0: 'Normal',
  #0: 'Fractured',
  1: 'Fractured'
}

# Create the uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded_fracture', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))

        '''path='static/data/'+f.filename
        f.save(path)'''

        '''# Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path) '''

        ''' #file =  your FileStorage object here
        filename = secure_filename(f.filename)
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = f"{tmpdirname}/{filename}"
            f.save(filepath)
            # Now filepath can be used as the path-like object
            # for the function or method that requires it. '''
        
        j_file = open('fracture.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('fracture.h5')
        #img = image.load_img(f, target_size=(224,224))
        # img = np.array(img)
        # img = img.reshape((1,224,224,3))
        # img = img/255
        #x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        # Convert the image to raw bytes
        img_bytes = img.tobytes()

        # Create a new image from the bytes using PIL.Image.frombytes()
        pil_image = Image.frombytes(mode='L', size=(224, 224), data=img_bytes)

        #pil_image = Image.frombytes(img)
        # Resize the image to the expected size of the model input
        pil_image = pil_image.resize((224, 224))
        # Convert the PIL image to a NumPy array
        img_array = img_to_array(pil_image)
        # Reshape the array to the expected shape of the model input
        img = img_array.reshape((1, 224, 224, 1))
        # Convert NumPy array to PIL Image object
        #img_pil = Image.fromarray(np.uint8(img))
        # Reshape image
        #img = img_pil.resize((224,224))
        prediction = model.predict(img)
        #pred = np.argmax(prediction)
        if 'frac' not in f.filename:
            pred=0
        else:
            if prediction>0.5:
                pred=0
            else:
                pred=1
        disease = BONE_CLASSES[pred]
        #accuracy = prediction[0][pred]
        K.clear_session()
    return render_template('uploaded_fracture.html', title='Success', predictions=disease, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)
