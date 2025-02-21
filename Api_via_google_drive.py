import tensorflow as tf
from flask import Flask,request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown


FILE_ID = "1arkE3-TflCxeO39In8oeq2YoqvogsvZ-"  
MODEL_PATH = "model_90.h5"

#downloading the model form google drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        print("Download complete.")   

download_model()

#load the model
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def upload_form():
    return '''
    <h1>Upload an image to classify as Dog or Cat</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">

        <input type="file" name="file" required>
        <button type="submit">Classify</button>
    </form>
    '''

# def preprocess_image(img): # for curl
#     img = img.resize((256,256))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array,axis = 0)
#     return img_array

def preprocess_image(img_path):
    img = image.load_img(img_path)
    img = img.resize((256,256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array,axis = 0)
    return img_array

@app.route('/predict',methods=['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no File provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    #save the image temorarily
    img_path = os.path.join('uploads',file.filename)
    file.save(img_path)

    # Read and preprocess the image(for curl)
    # img = Image.open(io.BytesIO(file.read()))
    # img_array = preprocess_image(img)

    img_array = preprocess_image(img_path)

    # Predict using the loaded model
    prediction = model.predict(img_array)

    result = 'dog' if prediction[0] > 0.5 else 'cat'

    os.remove(img_path)
    # return f"Prediction:{result}"

    return jsonify({"prediction":result}),200 #for curl

if __name__ == '__main__':
    app.run(debug=True)


