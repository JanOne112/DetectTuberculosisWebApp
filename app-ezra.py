from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Definisikan layer kustom FixedDropout
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        return super(FixedDropout, self).call(inputs, training=training)

# Muat model dengan custom_objects
try:
    model = load_model('models/effnetb3-80.h5', custom_objects={'FixedDropout': FixedDropout})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Definisikan nama kelas
class_names = [
    'Normal',                 # 0
    'Tuberculosis',           # 1
    'Not a valid x-ray image' # 2
]

def predict_label(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 'Not a valid x-ray image'
    
    # Resize gambar sesuai ukuran input model (misalnya 224x224)
    resized = cv2.resize(img, (224, 224))
    
    # Preprocess gambar
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Lakukan prediksi
    yhat = model.predict(img_array, verbose=False)
    
    # Asumsikan model mengeluarkan probabilitas untuk setiap kelas
    predicted_class = np.argmax(yhat, axis=1)[0]
    
    return class_names[predicted_class]

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        name = request.form.get('name')
        age = request.form.get('age')

        if img:
            filename = secure_filename(img.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)

            prediction = predict_label(img_path)

            # Buat pesan untuk pengguna berdasarkan prediksi
            message = ''
            if prediction == 'Normal':
                message = f"Hello {name}, at the age of {age}, it appears that your lungs are normal."
            elif prediction == 'Tuberculosis':
                message = f"Hello {name}, at the age of {age}, you may have Tuberculosis. Please consult a healthcare professional."
            else:
                message = f"Hello {name}, it seems that the image is not a valid x-ray image for detection."

            # Path relatif untuk ditampilkan di template
            img_path_display = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template("classification.html", prediction=prediction, img_path=img_path_display, message=message)
        else:
            message = "No image uploaded."
            return render_template("classification.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)
