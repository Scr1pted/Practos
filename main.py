import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(88, 67, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (5,5), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2,  activation='softmax')
])

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (88, 67))
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels
bad = r'W:\uyyu\0'
good = r'W:\uyyu\1'

image, labels = load_images_from_folder(bad, 0)
image2, labels2 = load_images_from_folder(good, 1)

images = image+image2
label = labels+labels2

laba = np.array(label)
ima = np.array(images)



x_train, x_test, y_train, y_test = train_test_split(ima, laba, test_size=0.2, random_state=42)


x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

y_train_cat = keras.utils.to_categorical(y_train, len(np.unique(laba)))
y_test_cat = keras.utils.to_categorical(y_test, len(np.unique(laba)))

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=1, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

print(model.summary())