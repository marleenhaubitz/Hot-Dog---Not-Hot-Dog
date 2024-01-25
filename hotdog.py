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


train_hotdogs_path = "Hot-Dog---Not-Hot-Dog/archive/train/hot_dog"
train_notdogs_path = "Hot-Dog---Not-Hot-Dog/archive/train/not_hot_dog"

test_hotdogs_path = "Hot-Dog---Not-Hot-Dog/archive/test/hot_dog"
test_notdogs_path = "Hot-Dog---Not-Hot-Dog/archive/test/not_hot_dog"






