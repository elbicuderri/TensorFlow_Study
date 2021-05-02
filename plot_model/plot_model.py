import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from statistics import mean

print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(True)

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
        
        self.batchnorm = BatchNormalization(trainable=True)
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
        self.batchnorm = BatchNormalization(trainable=True)
        self.relu = Activation('relu')
        
        self.block11 = ResidualBlock(16, 3, 'same', downsample=False)
        self.block12 = ResidualBlock(16, 3, 'same', downsample=False)
        self.block21 = ResidualBlock(32, 3, 'same', downsample=True)
        self.block22 = ResidualBlock(32, 3, 'same', downsample=False)
        self.block31 = ResidualBlock(64, 3, 'same', downsample=True)
        self.block32 = ResidualBlock(64, 3, 'same', downsample=False)
        
        self.avg_pool = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.fc = Dense(10)
        self.softmax = Activation('softmax')

    def call(self, x):
        out0 = self.conv0(x)
        out0 = self.batchnorm(out0)
        out0 = self.relu(out0)

        out11 = self.block11(out0)
        out12 = self.block12(out11)
        out21 = self.block21(out12)
        out22 = self.block22(out21)
        out31 = self.block31(out22)
        out32 = self.block32(out31) ## updated

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

plot_model(model.model(), to_file='model.png', show_shapes=True, show_layer_names=False)