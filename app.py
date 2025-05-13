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

# Create the uploads folder if it doesn't exist
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
    model = load_model('models/effnetb3-88.h5', custom_objects={'FixedDropout': FixedDropout})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Definisikan nama kelas yang benar
class_names = [
    'Tuberculosis',       # 0
    'Normal',             # 1
]


def predict_label(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 'Not a valid x-ray image'
    
    
    # Resize gambar sesuai ukuran input model (misalnya 224x224)
    resized = cv2.resize(img, (224, 224))
    
    
    # Preprocess gambar (normalisasi)
    img_array = np.array(resized) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Lakukan prediksi
    yhat = model.predict(img_array, verbose=False)
    yhat_value = float(yhat[0])  # Ambil nilai prediksi sebagai float
    print(f"Prediction: {yhat_value}")  # Print untuk melihat hasil yhat

    # Logika prediksi berdasarkan nilai yhat
        
    # if yhat >= 0.5:
    #     return class_names[1]
    # else:
    #     return class_names[0]
    
    if yhat_value >= 0.5 and yhat_value <= 1.0:
        return class_names[1]
    else:
        return class_names[0]

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        name = request.form.get('name')
        age = request.form.get('age')

        if img:
            filename = secure_filename(img.filename)
            img_path = "static/" + filename
            img.save(img_path)

            # prediction = model.predict(img_path)
            # predict_class = np.argmax(prediction[0])
            # predict_label = class_names[predict_class]
            # accuracy = prediction[0][predict_class] * 100
            # accuracy_formatted = "{:.2f}".format(accuracy)

            prediction = predict_label(img_path)

              # Mengirimkan hasil klasifikasi ke halaman web
        #     return render_template('classification.html', label=predict_label, accuracy=accuracy_formatted, image_path=img_path)
        # else:
        #     return render_template('classification.html', label=None, accuracy=None, image_path=None)

            # message = ''
            # if prediction == 'Normal':
            #     message = f"Hello {name}, at the age of {age}, it appears that your lungs are normal."
            # elif prediction == 'Tuberculosis':
            #     message = f"Hello {name}, at the age of {age}, you may have Tuberculosis. Please consult a healthcare professional."
            # else:
            #     message = f"Hello {name}, it seems that the image is not a valid x-ray image for detection."

            # Buat pesan untuk pengguna berdasarkan prediksi
            message = ''
            if prediction == 'Normal':
                message = f"Hello {name}, at the age of {age}, it appears that your lungs are normal."
            elif prediction == 'Tuberculosis':
                message = f"Hello {name}, at the age of {age}, you may have Tuberculosis. Please consult a healthcare professional."
            else:
                message = f"Hello {name}, it seems that the image is not a valid x-ray image for detection."

            return render_template("classification.html", prediction=prediction, img_path=img_path, message=message)
        else:
            message = "No image uploaded."
            return render_template("classification.html", accuracy=None, message=message)

if __name__ == '__main__':
    app.run(debug=True)
