# Neural Net Hotdog Image Recognition

# Neural net docs 
# Installing Tensorflow is necessary, apparently. 
# https://keras.io/api/ 
# https://keras.io/getting_started/intro_to_keras_for_engineers/

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print("Keras version is: " + keras.__version__)



#Load Training Data
directory = "archive/train"
train_ds = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
)

#Load Test Data
directory2 = "archive/test"
test_ds = keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
)

#Show the Training Images
images_1 = plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

plt.show() 

#Show the Test Images 
images_2 = plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

plt.show() 









