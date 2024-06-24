from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load the pre-trained models
disease_model = load_model('Models\disease_detection\densenet121.h5')
severity_model = load_model('Models\severity_calculation\severity_model.h5')


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
    'sheath_blight': 8,
    'tungro': 9
}

index_to_class = {v: k for k, v in class_indices.items()}
disease_severity_dict={0: 'leaf_scald_severe', 1: 'bacterial_leaf_blight_severe', 2: 'leaf_scald_mild', 3: 'brown_spot_mild', 4: 'sheath_blight_mild', 5: 'leaf_blast_severe', 6: 'neck_blast_severe', 7: 'narrow_brown_spot_mild', 8: 'neck_blast_mild', 9: 'rice_hispa_mild', 10: 'tungro_mild', 11: 'rice_hispa_severe', 12: 'brown_spot_severe', 13: 'leaf_blast_mild', 14: 'tungro_severe', 15: 'sheath_blight_severe', 16: 'bacterial_leaf_blight_mild', 17: 'narrow_brown_spot_severe'}


# Prediction function
def predict_image(model, img_path, index_to_class):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class_name = index_to_class[predicted_index]
    return predicted_class_name, prediction

# Severity prediction function
def predict_severity(img_path, severity_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = severity_model.predict(img_array)
    predicted_index = np.argmax(prediction)
    severity_label = disease_severity_dict[predicted_index]
    return severity_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            predicted_class_name, probabilities = predict_image(disease_model, filename, index_to_class)
            if predicted_class_name == 'healthy':
                severity = 'N/A'
            else:
                severity = predict_severity(filename, severity_model)
            return render_template('index.html', filename=file.filename, index_to_class=index_to_class, prediction=predicted_class_name, probabilities=probabilities[0], severity=severity, enumerate=enumerate)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
