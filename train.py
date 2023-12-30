import os
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("train\\fer2013.csv")

label_to_text = {
    0: "Bravo",
    1: "Desgosto",
    2: "Medo",
    3: "Feliz",
    4: "Triste",
    5: "Surpresa",
    6: "Neutro",
}


img_array = df.pixels.apply(
    lambda x: np.array(x.split(" ")).reshape(48, 48, 1).astype("float32")
)

img_array = np.stack(img_array, axis=0)

labels = df.emotion.values
X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size=0.2)

X_train = X_train / 255
X_test = X_test / 255

basemodel = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(7, activation="softmax"),
    ]
)

basemodel.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

try:
    os.mkdir("checkpoint")
except FileExistsError:
    pass

file_name = "best_model.h5"
checkpoint_path = os.path.join("checkpoint", file_name)

call_back = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_freq="epoch",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
)

basemodel.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[call_back])


final_model = tf.keras.models.load_model(checkpoint_path)

from IPython.display import clear_output
import time

for k in range(40):
    print(f"Label atual é {label_to_text[y_test[k]]}")
    predicted_class = final_model.predict(tf.expand_dims(X_test[k], 0)).argmax()
    print(f"Label previsto é {label_to_text[predicted_class]}")
    plt.imshow(X_test[k].reshape((48, 48)))
    plt.show()
    time.sleep(3)
    clear_output(wait=True)