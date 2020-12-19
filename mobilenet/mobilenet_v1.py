import tensorflow as tf 

class MoblieNetV1(tf.keras.Model):
    def __init__(self, in_channels, n_classes):
        super(MoblieNetV1, self).__init__()
        
        def depthwise_seperable_conv(x):
            out = DepthwiseConv2D()(out)
            out = BatchNormalization()(out)
            out = Activation('relu')(out)
            
            out = Conv2D(kernel_size=1)(out)
            out = BatchNormalization()(out)
            out = Activation('relu')(out)
            
            return out
        
    def call(self, x):
        out = depthwise_seperable_conv(x)
        