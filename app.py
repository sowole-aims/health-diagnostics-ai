from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import zipfile
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shutil
import tempfile
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths and directories
UPLOAD_FOLDER_TB = 'datasets/TB/'
UPLOAD_FOLDER_NORMAL = 'datasets/Normal/'
MODEL_PATH = 'model/tb_cnn_model.keras'
HISTORY_FOLDER = 'static/history/'

os.makedirs(UPLOAD_FOLDER_TB, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_NORMAL, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.ERROR, 
                    format='%(asctime)s - %(message)s')

def extract_images_from_zip(zip_file):
    try:
        # Temporary directory to hold extracted files
        temp_dir = 'temp_extracted/'
        os.makedirs(temp_dir, exist_ok=True)

        # Extract files to temp directory
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(temp_dir)

        # Loop through extracted files
        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                full_temp_path = os.path.join(root, file_name)
                
                # Check if the filename indicates TB or Normal
                if 'tb' in file_name.lower() or 'tuberculosis' in file_name.lower():
                    destination_folder = UPLOAD_FOLDER_TB
                    logging.info(f"Identified as TB image: {file_name}")
                elif 'normal' in file_name.lower():
                    destination_folder = UPLOAD_FOLDER_NORMAL
                    logging.info(f"Identified as Normal image: {file_name}")
                else:
                    # Skip files that don't match any criteria
                    logging.warning(f"File '{file_name}' does not match 'TB' or 'Normal'. Skipping.")
                    continue

                # Define destination path and move file
                destination_path = os.path.join(destination_folder, file_name)
                shutil.move(full_temp_path, destination_path)
                logging.info(f"Moved '{file_name}' to '{destination_folder}'")

        # Clean up temporary directory after processing
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logging.error(f"Failed to extract and move images: {str(e)}")




def process_uploaded_file(file):
    if file.filename.endswith('.zip'):
        # Use a temporary file for the ZIP to avoid Windows locking issues
        with tempfile.NamedTemporaryFile(delete=False) as temp_zip:
            file.save(temp_zip.name)
            temp_zip_path = temp_zip.name

        # Now extract images from the temp file and remove it afterward
        extract_images_from_zip(temp_zip_path)
        os.remove(temp_zip_path)
    else:
        logging.warning("Uploaded file is not a ZIP file.")


def count_images_in_folder(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def load_data(balance_data=False, split_ratio=0.8):
    images, labels = [], []

    tb_count = count_images_in_folder(UPLOAD_FOLDER_TB)
    normal_count = count_images_in_folder(UPLOAD_FOLDER_NORMAL)

    if tb_count == 0 or normal_count == 0:
        raise ValueError("No images found in one or both folders. Please upload valid images.")

    for filename in os.listdir(UPLOAD_FOLDER_TB):
        try:
            image = Image.open(os.path.join(UPLOAD_FOLDER_TB, filename)).convert('L')
            images.append(np.array(image.resize((256, 256))) / 255.0)
            labels.append(1)
        except UnidentifiedImageError:
            logging.error(f"Invalid image: {filename}")

    for filename in os.listdir(UPLOAD_FOLDER_NORMAL):
        try:
            image = Image.open(os.path.join(UPLOAD_FOLDER_NORMAL, filename)).convert('L')
            images.append(np.array(image.resize((256, 256))) / 255.0)
            labels.append(0)
        except UnidentifiedImageError:
            logging.error(f"Invalid image: {filename}")

    images = np.array(images).reshape(-1, 256, 256, 1)
    labels = np.array(labels)

    if balance_data:
        smote = SMOTE()
        images, labels = smote.fit_resample(images.reshape(len(images), -1), labels)
        images = images.reshape(-1, 256, 256, 1)

    return train_test_split(images, labels, test_size=1 - split_ratio, random_state=42)

def build_model(architecture):
    if architecture == "3-Layer CNN":
        return Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    else:
        return Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plot_path = os.path.join(HISTORY_FOLDER, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    file = request.files['image_files']  # Update to match the HTML form input name
    process_uploaded_file(file)  # Process the uploaded file
    flash('Images uploaded successfully.')
    return redirect(url_for('home'))

@app.route('/check_balance')
def check_balance():
    tb_count = count_images_in_folder(UPLOAD_FOLDER_TB)
    normal_count = count_images_in_folder(UPLOAD_FOLDER_NORMAL)

    if tb_count == normal_count:
        message = "The dataset is balanced."
    else:
        message = f"The dataset is imbalanced. TB: {tb_count}, Normal: {normal_count}"

    return render_template('balance.html', message=message)

@app.route('/configure', methods=['GET', 'POST'])
def configure_model():
    if request.method == 'POST':
        session['architecture'] = request.form['architecture']
        session['epochs'] = int(request.form['epochs'])
        session['batch_size'] = int(request.form['batch_size'])
        session['split_ratio'] = float(request.form['split_ratio'])
        session['balance_option'] = request.form['balance_option']
        return redirect(url_for('train_model'))
    return render_template('configure_model.html')

@app.route('/train')
def train_model():
    X_train, X_test, y_train, y_test = load_data(
        balance_data=session.get('balance_option') == 'SMOTE Oversampling',
        split_ratio=session.get('split_ratio', 0.8)
    )

    model = build_model(session.get('architecture', '3-Layer CNN'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=session.get('epochs', 10),
                        batch_size=session.get('batch_size', 16),
                        validation_data=(X_test, y_test))

    model.save(MODEL_PATH)

    train_accuracy = history.history['accuracy'][-1] 
    val_accuracy = history.history['val_accuracy'][-1] 

    plot_path = plot_training_history(history.history)

    return render_template(
        'train.html',
        plot_path=plot_path,
        train_accuracy=f"{train_accuracy:.2%}",
        val_accuracy=f"{val_accuracy:.2%}"
    )

@app.route('/evaluate')
def evaluate_model():
    model = load_model(MODEL_PATH)
    _, X_test, _, y_test = load_data()
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = classification_report(y_test, y_pred, target_names=["Normal", "TB"])
    return render_template('evaluate.html', report=report)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'xray' not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        
        uploaded_file = request.files['xray']
        
        if uploaded_file.filename == '':
            flash("No file selected for uploading.")
            return redirect(request.url)

        # Save the uploaded file temporarily
        file_path = os.path.join('datasets/', uploaded_file.filename)
        uploaded_file.save(file_path)
        
        # Load model and make a prediction
        model = load_model(MODEL_PATH)
        image = Image.open(file_path).convert('L').resize((256, 256))
        image = np.array(image) / 255.0
        image = image.reshape(-1, 256, 256, 1)

        prediction = model.predict(image)
        result = "TB" if prediction[0][0] > 0.5 else "Normal"
        confidence = round(float(prediction[0][0] * 100), 2) if result == "TB" else round(float((1 - prediction[0][0]) * 100), 2)
        
        # Clean up by removing the file
        os.remove(file_path)
        
        return render_template('predict.html', result=result, confidence=confidence)
    
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
