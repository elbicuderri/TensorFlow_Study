import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib
import statistics
#import tqdm

print(device_lib.list_local_devices())

# tf.debugging.set_log_device_placement(True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess(x, y):
    image = tf.reshape(x, [32, 32, 3])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 127.5) / 127.5
    label = tf.one_hot(y, depth=10)
    label = tf.squeeze(label)
    return image, label

train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(3).batch(32)

valid_loader = valid_dataset.map(preprocess).repeat(3).batch(32)

# train_loader = train_dataset.batch(32)

# for img, lbl in train_loader.take(1):
#     print(img.shape)
#     print(lbl.shape)

def conv33(filters, kernel_size, strides, padding='same', use_bias=False):
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=use_bias
    )

# def convBlock():

class SimpleResNet(tf.keras.Model):
    def __init__(self):
        super(SimpleResNet, self).__init__()
    
        # self.conv0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu')
        # ])
        
        self.trainable = True

        self.conv = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization(trainable=self.trainable)

        self.relu = tf.keras.layers.Activation('relu')
    
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10, activation='softmax')
        # self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, x):
        # out0 = self.conv0(x)
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = self.flatten(out3)
        out5 = self.fc(out4)
        # out6 = self.softmax(out5)
       
        return out5

model = SimpleResNet()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

batch_size = 32
epochs = 1

# train_step = len(train_loader)
# valid_step = len(valid_loader)

loss_dict = {}
val_loss_dict = {}

# with tf.device('/GPU:0'):
for epoch in range(1, epochs + 1):
    
    loss_list = []   
    for i, (img, label) in enumerate(train_loader):
        model.trainable = True
        print(model.batchnorm.moving_mean.trainable)
        model_params = model.trainable_variables
        if (i == 0):
            print(model_params)

        with tf.GradientTape() as tape:
            out = model(img)
            
            loss = loss_fn(out, label)

            loss_list.append(loss.numpy().sum())

            # print(loss.numpy().sum())

        grads = tape.gradient(loss, model_params)

        optimizer.apply_gradients(zip(grads, model_params))
    
    loss_dict[epoch] = statistics.mean(loss_list)
    print(f"[{epoch}/{epochs}] : {loss_dict[epoch]}")
        
    
    val_loss_list = []
    for j, (val_img, val_label) in enumerate(valid_loader):
        model.trainable = False
        if (j == 0):
            print(model.trainable_variables)
        
        val_out = model(val_img)
        
        val_loss = loss_fn(val_out, val_label)
        
        val_loss_list.append(val_loss.numpy().sum())

        # print(val_loss.numpy().sum())
        
    val_loss_dict[epoch] = statistics.mean(val_loss_list)
    print(f"[{epoch}/{epochs}] : {val_loss_dict[epoch]}")

    print(f"[{epoch}/{epochs}] finished")
    print('==================')

    # if (epoch == 5):
    #     model.save_weights('model/cifar10_model_5', save_format='tf')

model.save_weights('model/cifar10_model', save_format='tf')

print('model saved')
