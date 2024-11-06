### Technical Documentation for AI-Powered Health Diagnostic Application


#### Overview
This application, built using Flask, provides an AI-powered diagnostic tool for classifying disease images, specifically tuberculosis (TB) versus normal cases. The platform allows users to upload a zip file containing TB and normal images, train a machine learning model on these images, check dataset balance, configure training settings, evaluate the model, and predict outcomes for new images. It uses Convolutional Neural Networks (CNNs) for image classification, leveraging TensorFlow and Keras.


### Table of Contents

1. **Environment Setup**
2. **File Structure**
3. **Application Workflow**
4. **Endpoints and Functionalities**
5. **Utility Functions**
6. **Key ML Components**
7. **Error Handling and Logging**
8. **Data and Model Storage**
9. **Instructions for Running the Application**
10. **Future Improvements**



### 1. Environment Setup

To ensure compatibility, set up a virtual environment and install the required libraries as listed below.

#### Required Libraries
- `Flask` for the web server.
- `tensorflow` for deep learning.
- `numpy`, `pandas` for data handling.
- `scikit-learn` for data splitting and evaluation metrics.
- `imblearn` for SMOTE (Synthetic Minority Over-sampling Technique).
- `matplotlib` for training history visualization.
- `Pillow` for image processing.

Create a `requirements.txt` file with the dependencies for easy installation.

#### Installation
```bash
pip install -r requirements.txt
```


### 2. File Structure

```
app/
│
├── app.py                 # Main application code
├── templates/             # HTML templates for each page
│   ├── index.html         # Home page
│   ├── balance.html       # Check dataset balance page
│   ├── configure_model.html  # Model configuration page
│   ├── train.html         # Model training results page
│   ├── evaluate.html      # Model evaluation page
│   └── predict.html       # Prediction page
├── datasets/              # Folder to store uploaded images for TB and normal
│   ├── TB/                # TB images folder
│   └── Normal/            # Normal images folder
├── model/                 # Stores the trained model
└── static/history/        # Stores the training history plots
```


### 3. Application Workflow

1. **Upload Zip File**: Users upload a zip file containing labeled TB and normal images.
2. **Extract and Organize Images**: The app organizes images based on file names to distinguish between TB and normal cases.
3. **Check Dataset Balance**: Users check if the dataset is balanced in terms of TB and normal images.
4. **Model Configuration and Training**: Users configure model parameters (architecture, epochs, batch size, data split ratio, balancing strategy) and train the model.
5. **Model Evaluation**: Post-training, the app displays model performance metrics (accuracy, precision, recall) and a plot of training history.
6. **Prediction**: Users can upload a new image to classify as TB or normal, with a confidence score.


### 4. Endpoints and Functionalities

#### `/`
**Method**: `GET`  
**Description**: Displays the home page, providing links to upload images, check dataset balance, configure and train model, evaluate, and predict.

#### `/upload`
**Method**: `POST`  
**Description**: Accepts a zip file upload. Calls `process_uploaded_file()` to extract and store TB and normal images in separate folders.

#### `/check_balance`
**Method**: `GET`  
**Description**: Counts images in TB and normal folders to determine dataset balance, displaying the result on the page.

#### `/configure`
**Method**: `GET`, `POST`  
**Description**: Configures model settings such as architecture, number of epochs, batch size, data split ratio, and balancing strategy.

#### `/train`
**Method**: `GET`  
**Description**: Trains the CNN model based on configured settings. Generates training accuracy, validation accuracy, and a plot of training history, saving the trained model.

#### `/evaluate`
**Method**: `GET`  
**Description**: Loads the saved model, evaluates it on test data, and displays a classification report (precision, recall, F1-score).

#### `/predict`
**Method**: `GET`, `POST`  
**Description**: Accepts an image upload, processes it, and uses the trained model to classify the image as TB or normal, displaying a confidence score.


### 5. Utility Functions

#### `extract_images_from_zip(zip_file)`
- **Purpose**: Extracts images from a zip file and moves them to appropriate folders based on file naming conventions.
- **Logic**: Detects TB or normal images by keywords in filenames and organizes images into respective folders.

#### `process_uploaded_file(file)`
- **Purpose**: Manages the upload process, ensuring that only zip files are accepted.
- **Logic**: Saves the uploaded file temporarily, calls `extract_images_from_zip()` to extract contents, and deletes the temp file after processing.

#### `count_images_in_folder(folder)`
- **Purpose**: Counts valid image files in a specified folder.
- **Logic**: Counts files with `.jpg`, `.jpeg`, or `.png` extensions.

#### `load_data(balance_data=False, split_ratio=0.8)`
- **Purpose**: Loads, preprocesses, and splits the dataset.
- **Logic**: Balances the dataset if `balance_data=True`, splits the data into train/test based on `split_ratio`, and returns split datasets.

#### `plot_training_history(history)`
- **Purpose**: Generates and saves a plot of training vs. validation accuracy and loss.
- **Logic**: Uses Matplotlib to create subplots for training/validation loss and accuracy, then saves the plot in `HISTORY_FOLDER`.



### 6. Key ML Components

#### Model Architecture
- **Function**: `build_model(architecture)`
- **Description**: Creates and compiles a CNN model based on the chosen architecture (3-layer or 2-layer CNN).
  
#### Model Training
- **Function**: `train_model()`
- **Process**:
  - Uses the settings defined by the user to train the CNN model.
  - Saves the model to `MODEL_PATH` for future use.

#### Model Evaluation
- **Function**: `evaluate_model()`
- **Process**:
  - Loads the saved model, evaluates on test data, and generates a classification report.
  - Presents accuracy, precision, recall, and F1 score for both TB and normal classifications.

#### Prediction
- **Function**: `predict()`
- **Process**:
  - Accepts a new image upload, processes it, and uses the model to classify the image.
  - Provides a prediction with a confidence score.



### 7. Error Handling and Logging

- **Logging**: Logs errors (e.g., invalid image files) in `app.log` and warns of unsupported file formats or extraction issues.
- **User Feedback**: Uses `flash()` to provide real-time user feedback for errors (e.g., invalid file format).


### 8. Data and Model Storage

- **Images**: Stored in `datasets/TB` and `datasets/Normal` folders.
- **Model**: Saved as `tb_cnn_model.keras` in the `model` directory.
- **Training History**: Saved plots for training and validation loss/accuracy are stored in `static/history/`.

---

### 9. Instructions for Running the Application

1. **Run the Application**:
   ```bash
   python app.py
   ```
   The application runs on `localhost:5000` by default.

2. **Uploading Images**:
   - Navigate to the homepage and upload a zip file with TB and normal images.
   - Ensure that images are labeled to indicate class (e.g., "tb" or "normal" in filenames).

3. **Training the Model**:
   - Configure model settings (architecture, epochs, batch size, etc.).
   - Start the training process and review the training results.

4. **Evaluating and Predicting**:
   - Evaluate model performance to generate a classification report.
   - Use the prediction feature to classify new images as TB or normal.



### 10. Future Improvements

- **Additional Model Architectures**: Incorporate more complex CNN architectures or transfer learning.
- **Enhanced Dataset Management**: Allow for more complex file/folder structures or automated data augmentation.
- **Improved Error Handling**: Detect incorrect file structures earlier and provide detailed error messages.
