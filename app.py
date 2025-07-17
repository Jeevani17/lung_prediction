from flask import Flask, render_template, request, flash, redirect, url_for
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Using dummy models for demonstration.")
    TENSORFLOW_AVAILABLE = False
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models with error handling
def load_models():
    models = {}
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow not available. Models will not be loaded.")
        return models
        
    try:
        if os.path.exists('pneumonia-detection-model.h5'):
            models['pneumonia'] = load_model('pneumonia-detection-model.h5')
            print("✅ Pneumonia model loaded successfully")
        else:
            print("⚠️ Pneumonia model not found")
            
        if os.path.exists('cancer-detection-model.h5'):
            models['cancer'] = load_model('cancer-detection-model.h5')
            print("✅ Cancer model loaded successfully")
        else:
            print("⚠️ Cancer model not found")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        
    return models

# Initialize models
models = load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path, model_type):
    try:
        if not TENSORFLOW_AVAILABLE:
            # Return dummy prediction for demonstration
            import random
            confidence = random.uniform(60, 95)
            if model_type == "pneumonia":
                label = "PNEUMONIA DETECTED" if confidence > 75 else "NORMAL"
            elif model_type == "cancer":
                label = "CANCER DETECTED" if confidence > 75 else "NO CANCER DETECTED"
            else:
                label = "UNKNOWN"
            return label, confidence
            
        if model_type not in models:
            return "Model not available", 0.0
            
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = models[model_type].predict(img_array)
        confidence = float(prediction[0][0]) * 100

        if model_type == "pneumonia":
            label = "PNEUMONIA DETECTED" if confidence > 50 else "NORMAL"
        elif model_type == "cancer":
            label = "CANCER DETECTED" if confidence > 50 else "NO CANCER DETECTED"
        else:
            label = "UNKNOWN"
            confidence = 0.0

        return label, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        prediction = ""
        confidence = None
        disease_type = ""
        image_url = ""
        error_message = ""

        if request.method == 'POST':
            disease_type = request.form.get('disease')
            file = request.files.get('file')
            
            if not file or file.filename == '':
                error_message = "Please select a file"
            elif not allowed_file(file.filename):
                error_message = "Please upload a valid image file (PNG, JPG, JPEG, GIF)"
            elif not disease_type:
                error_message = "Please select a disease type"
            elif disease_type not in models:
                error_message = f"Model for {disease_type} detection is not available"
            else:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    prediction, confidence = predict_disease(filepath, disease_type)
                    image_url = filepath
                    
                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"

        return render_template('index.html',
                             prediction=prediction,
                             confidence=confidence,
                             disease_type=disease_type,
                             image_url=image_url,
                             error_message=error_message,
                             models_available=list(models.keys()))
                             
    except Exception as e:
        return f"Application Error: {e}", 500

@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'upload_folder': app.config['UPLOAD_FOLDER']
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)