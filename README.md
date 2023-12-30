Link csv.
https://drive.google.com/file/d/1-3Buxxb25tqZb1b9qyHXXNi2brqr6R58/view?usp=sharing

# README.md for train.py
This script (train.py) is designed to train a Convolutional Neural Network (CNN) for emotion recognition using the FER2013 dataset. The FER2013 dataset contains facial expressions labeled with different emotions, including happiness, sadness, fear, disgust, surprise, and neutral.

Prerequisites
Before running the script, make sure you have the following installed:

Python (3.6 or later)
TensorFlow
pandas
matplotlib
scikit-learn
You can install the required Python packages using the following command:

```bash
pip install tensorflow pandas matplotlib scikit-learn
```
Usage
Download the FER2013 dataset from Kaggle.
Place the fer2013.csv file in the train/ directory.
Script Execution
Run the train.py script using the following command:

```bash
python train.py
```

Script Overview
Data Loading and Preprocessing:

Loads the FER2013 dataset from the train/fer2013.csv file.
Maps emotion labels to corresponding text descriptions.
Data Splitting:

Splits the dataset into training and testing sets.
Data Normalization:

Normalizes pixel values to the range [0, 1].
Model Architecture:

Constructs a CNN model using TensorFlow's Keras API.
Configures the model with convolutional layers, max-pooling, batch normalization, flattening, and dense layers.
Model Compilation:

Compiles the model with the RMSprop optimizer and sparse categorical crossentropy loss.
Model Training:

Trains the model on the training set for 50 epochs.
Performs validation on a subset of the training data.
Saves the best-performing model during training in the checkpoint/best_model.h5 file.
Model Evaluation:

Loads the best-performing model.
Displays real-time predictions for a subset of test images with their actual and predicted labels using matplotlib.
Additional Notes
The script creates a checkpoint/ directory to store the best model during training.
Adjust hyperparameters or model architecture as needed.
Ensure that the dataset file (fer2013.csv) is correctly placed in the train/ directory.
Feel free to customize the script according to your requirements and experiment with different configurations for better performance.






# README.md for predict_emotion.py
This script (predict_emotion.py) utilizes a pre-trained Convolutional Neural Network (CNN) model to predict emotions in facial images. The model is assumed to be saved in the checkpoint/best_model.h5 file, and it should have been trained using the train.py script.

Prerequisites
Before running the script, make sure you have the following installed:

Python (3.6 or later)
TensorFlow
OpenCV
matplotlib
NumPy
You can install the required Python packages using the following command:

```bash
pip install tensorflow opencv-python matplotlib numpy
```	

Usage
Ensure that the pre-trained model (best_model.h5) is available in the checkpoint/ directory. You can train the model using the train.py script.
Place the images you want to predict in the images/ directory.
Script Execution
Run the predict_emotion.py script using the following command:

```bash
python predict_emotion.py
```

The script will process each image in the images/ directory, detect faces, and display the predicted emotion along with a bounding box around the detected face.

Script Overview
Model Loading:

Loads the pre-trained model from the checkpoint/best_model.h5 file.
Image Prediction:

Reads each image from the images/ directory.
Detects faces in the image using Haar cascades.
Resizes and preprocesses the face region.
Uses the pre-trained model to predict the emotion.
Displays the image with a bounding box around the detected face and the predicted emotion if confidence is above 70%.
Supported Image Formats:

The script supports images with the following extensions: .jpg, .jpeg, .png, .gif, .bmp.
Additional Notes
Adjust the image_folder_path variable to point to the directory containing the images you want to predict.
Ensure that the pre-trained model file (best_model.h5) is available in the checkpoint/ directory.
Make sure OpenCV is correctly installed and configured.
Feel free to customize the script or integrate it into your applications for real-time emotion prediction in facial images.