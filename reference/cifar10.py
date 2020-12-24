import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, concatenate
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
# from tensorflow.data.Dataset import from_tensor_slices
from tensorflow.python.client import device_lib
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_loader = train_dataset.batch(batch_size)

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

valid_loader = valid_dataset.batch(batch_size)

def conv33(filters, kernel_size, strides=1, padding='same', use_bias=False):
    return Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=use_bias
    )
    
class SimpleResNet(tf.keras.Model):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        
        # self.conv33 = conv33
        
        self.batchnorm = BatchNormalization(trainable=True)
        self.relu = Activation('relu')
        self.avg_pool = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.fc = Dense(10)
        self.softmax = Activation('softmax')
        
        # self.concatenate = Concatenate()
        
        # def conv_block(filters, kernel_size, padding='same',
        #                strides=1, use_bias=False):
            
        # out1 = (self.relu(self.batchnorm(self.conv33(16,3)(out0))))
        
    def call(self, x):
        out0 = conv33(filters=16, kernel_size=3, strides=1)(x)
        out0 = self.batchnorm(out0)
        out0 = self.relu(out0)
        
        out1 = conv33(filters=16, kernel_size=3, strides=1)(out0)
        out1 = self.batchnorm(out1)
        out1 = self.relu(out1)
        out1 = conv33(filters=16, kernel_size=3, strides=1)(out1)
        out1 = self.batchnorm(out1)
        
        out1 = conv33(filters=16, kernel_size=3, strides=1)(out1)
        out1 = self.batchnorm(out1)
        out1 = self.relu(out1)
        out1 = conv33(filters=16, kernel_size=3, strides=1)(out1)
        out1 = self.batchnorm(out1)
        
        res2 = conv33(filters=32, kernel_size=3, strides=2)(out1)
        
        out2 = conv33(filters=32, kernel_size=3, strides=2)(out1)
        out2 = self.batchnorm(out2)
        out2 = self.relu(out2)
        out2 = conv33(filters=32, kernel_size=3, strides=1)(out2)
        out2 = self.batchnorm(out2)
        
        out2 = conv33(filters=32, kernel_size=3, strides=1)(out2)
        out2 = self.batchnorm(out2)
        out2 = self.relu(out2)
        out2 = conv33(filters=32, kernel_size=3, strides=1)(out2)
        out2 = self.batchnorm(out2)
        
        # out2 = concatenate([out2, res2])
        out2 += res2
        out2 = self.relu(out2)
        
        res3 = conv33(filters=64, kernel_size=3, strides=2)(out2)
        
        out3 = conv33(filters=64, kernel_size=3, strides=2)(out2)
        out3 = self.batchnorm(out3)
        out3 = self.relu(out3)
        out3 = conv33(filters=64, kernel_size=3, strides=1)(out3)
        out3 = self.batchnorm(out3)
        
        out3 = conv33(filters=64, kernel_size=3, strides=1)(out3)
        out3 = self.batchnorm(out3)
        out3 = self.relu(out3)
        out3 = conv33(filters=64, kernel_size=3, strides=1)(out3)
        out3 = self.batchnorm(out3)
        
        # out3 = concatenate([out3, res3])
        out3 += res3
        out3 = self.relu(out3)
        
        out3 = self.avg_pool(out3)
        out3 = self.flatten(out3)
        
        out3 = self.fc(out3)
        out = self.softmax(out3)
          
        return out
    
    
model = SimpleResNet()

loss_fn = CategoricalCrossentropy()

optimizer = Adam(lr=1e-3)

batch_size = 32

epochs = 2
        
for epoch in range(1, epochs + 1):
    loss_list = []   

    for i, (img, label) in enumerate(train_loader):
        model_params = model.trainable_variables

        with tf.GradientTape() as tape:
            out = model(img)

            loss = loss_fn(out, label)

            # loss_list.append(loss.numpy().sum())

            # print(loss.numpy().sum())

        grads = tape.gradient(loss, model_params)

        optimizer.apply_gradients(zip(grads, model_params))

    # loss_dict[epoch] = statistics.mean(loss_list)

    # print(loss_dict[epoch])
    print(f"[{epoch}/{epochs}] finished")
    print('==================')
        
model.save_weights('cifar10_resnet', save_format='tf')
