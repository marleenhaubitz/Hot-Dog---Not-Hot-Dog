# Neural Net Hotdog Image Recognition

# Neural net docs 
# Installing Tensorflow is necessary, apparently. 
# https://keras.io/api/ 
# https://keras.io/getting_started/intro_to_keras_for_engineers/

import tensorflow as tf
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print("Keras version is: " + keras.__version__)

directory = "fill"

keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    data_format=None,
)







