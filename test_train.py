import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt


model_path = "checkpoint/best_model.h5"
trained_model = load_model(model_path)


image_folder_path = "images"


label_to_text = {
    0: "Bravo",
    1: "Desgosto",
    2: "Medo",
    3: "Feliz",
    4: "Triste",
    5: "Surpresa",
    6: "Neutro",
}


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_emotion(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print(f"No faces detected in: {image_path}")
        return

    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]

        if w < 30 or h < 30:
            print(f"Skipped small face in: {image_path}")
            continue

        face_roi_resized = cv2.resize(face_roi, (48, 48))

        face_roi_resized = face_roi_resized / 255.0

        predictions = trained_model.predict(np.expand_dims(face_roi_resized, axis=0))
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        if confidence > 70:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, color="cyan", linewidth=2, fill=False)
            )
            plt.title(
                f"Predicted: {label_to_text[predicted_class]} (Confidence: {confidence:.2f}%)"
            )
            plt.show()


for filename in os.listdir(image_folder_path):
    if any(
        filename.lower().endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    ):
        image_path = os.path.join(image_folder_path, filename)

        predict_emotion(image_path)
