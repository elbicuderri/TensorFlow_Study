import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import math
from tensorflow.keras.layers import *
from tensorflow import keras

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

@tf.function
def normalize(image, label):
    return image, label

@tf.function
def augment(image, label):
    return image, label

