from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os

app = Flask(__name__)

# Folder untuk menyimpan gambar yang diupload
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Keras model
model = load_model("model/mobilenet-ke2.h5")

# Define the class labels
class_labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Utility function to validate uploaded files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(uploaded_image):
    img = uploaded_image.resize((224, 224))  # Resize to match model input size
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    # Get confidence level (convert to float)
    confidence = float(predictions[0][predicted_class_index]) * 100  # Multiply by 100 for percentage

    return predicted_class, confidence

# Flask Routes
@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save uploaded file
        uploaded_image_filename = file.filename
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)
        file.save(uploaded_image_path)

        # Load and predict
        img = Image.open(uploaded_image_path)
        prediction, confidence = predict_image(img)

        # Annotate the image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Prediction: {prediction}", fill=(255, 0, 0), font=font)

        # Save annotated image
        annotated_image_filename = "annotated_" + uploaded_image_filename
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_image_filename)
        img.save(annotated_image_path)

        return render_template("index.html", 
                               prediction=prediction, 
                               confidence=confidence,
                               uploaded_image=uploaded_image_filename,  # Pass filename only
                               annotated_image=annotated_image_filename)  # Pass filename only
    else:
        return "Invalid file type", 400

if __name__ == '__main__':
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
