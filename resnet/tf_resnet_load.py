import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from statistics import mean

print(device_lib.list_local_devices())

# tf.debugging.set_log_device_placement(True)

def conv33(filters, kernel_size=3, strides=1, padding='same', use_bias=False):
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=use_bias
    )
    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, padding='same', downsample=True, use_bias=False):
        super(ResidualBlock, self).__init__()

        self.downsample = downsample

        if (self.downsample == True):
            self.conv1 = conv33(filters=out_channels,
                                kernel_size=kernel_size,
                                strides=2,
                                padding=padding,
                                use_bias=use_bias) ## downsample
        
        else:
            self.conv1 = conv33(filters=out_channels,
                                kernel_size=kernel_size,
                                strides=1,
                                padding=padding,
                                use_bias=use_bias) 
        
        self.conv2 = conv33(filters=out_channels,
                            kernel_size=kernel_size,
                            strides=1,
                            padding=padding,
                            use_bias=use_bias)
        
        self.batchnorm = BatchNormalization()
        self.relu = Activation('relu')
        
    def call(self, x):
        if (self.downsample == True):
            identity = self.conv1(x)
        
        else:
            identity = x

        out0 = self.conv1(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.conv2(out2)
        out4 = self.batchnorm(out3)
        out4 += identity
        out5 = self.relu(out4)
        
        return out5
        

class SimpleResNet(tf.keras.Model):
    def __init__(self):
        super(SimpleResNet, self).__init__()
    
        self.conv0 = conv33(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False)
        self.batchnorm = BatchNormalization()
        self.relu = Activation('relu')
        self.avg_pool = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.fc = Dense(10)
        self.softmax = Activation('softmax')

    def call(self, x):
        out0 = self.conv0(x)
        out0 = self.batchnorm(out0)
        out0 = self.relu(out0)

        out11 = ResidualBlock(16, 3, 'same', downsample=False)(out0)
        out12 = ResidualBlock(16, 3, 'same', downsample=False)(out11)
        out21 = ResidualBlock(32, 3, 'same', downsample=True)(out12)
        out22 = ResidualBlock(32, 3, 'same', downsample=False)(out21)
        out31 = ResidualBlock(64, 3, 'same', downsample=True)(out22)
        out32 = ResidualBlock(64, 3, 'same', downsample=False)(out31)

        out4 = self.avg_pool(out32)
        out5 = self.flatten(out4)
        out6 = self.fc(out5)
        out = self.softmax(out6)
       
        return out
    
    def model(self):
        inputs = tf.keras.Input(shape=(32, 32, 3))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

model = SimpleResNet()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype("float32")
x_test = x_test / 255.0
x_test = (x_test - 0.5) / 0.5
# y_test = to_categorical(y_test)

test_data = x_test[:32, :, :, :]

out = model(test_data)

print(out.shape)

checkpoint_dir = "model/"

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model.load_weights(latest)

weights = model.get_weights()

# print(weights)

print(dir(model))

# print(model.get_layer[0])

# print(model.get_input_at())

# print(model.input_shape())

print(model.trainable_variables)

print(len(model.trainable_variables))

print(len(model.trainable_weights))

## something is wrong....


