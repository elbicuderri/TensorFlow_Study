import tensorflow as tf
from tensorflow.keras.layers import *

cifar = tf.random.uniform([1, 32, 32, 3], maxval=1)

out1 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=1,
                      padding='valid')(cifar)

out2 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=1,
                      padding='same')(cifar)

out3 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=2,
                      padding='valid')(cifar)

out4 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=2,
                      padding='same')(cifar)

print(out1.shape)

print(out2.shape)

print(out3.shape)

print(out4.shape)