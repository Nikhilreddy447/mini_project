from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('Models\disease_detection\densenet121.h5')

# Class indices mapping
class_indices = {
    'bacterial_leaf_blight': 0,
    'brown_spot': 1,
    'healthy': 2,
    'leaf_blast': 3,
    'leaf_scald': 4,
    'narrow_brown_spot': 5,
    'neck_blast': 6,
    'rice_hispa': 7,
    'shelth_blight': 8,
    'tungro': 9}
index_to_class = {v: k for k, v in class_indices.items()}

# Prediction function
def predict_image(model, img_path, index_to_class):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class_name = index_to_class[predicted_index]
    return predicted_class_name, prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            predicted_class_name, probabilities = predict_image(model, filename, index_to_class)
            return render_template('index.html', filename=file.filename,index_to_class=index_to_class, prediction=predicted_class_name, probabilities=probabilities[0],enumerate=enumerate)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
