import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import *
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
        
        # out11 = ResidualBlock(16, 3, 'same', downsample=False)(out0)
        # out12 = ResidualBlock(16, 3, 'same', downsample=False)(out11)
        # out21 = ResidualBlock(32, 3, 'same', downsample=True)(out12)
        # out22 = ResidualBlock(32, 3, 'same', downsample=False)(out21)
        # out31 = ResidualBlock(64, 3, 'same', downsample=True)(out22)
        # out32 = ResidualBlock(64, 3, 'same', downsample=False)(out31)

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

model.model().summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess(x, y):
    image = tf.reshape(x, [32, 32, 3])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5

    label = tf.one_hot(y, depth=10)
    label = tf.squeeze(label)

    return image, label

batch_size = 32
epochs = 5

# train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(epochs).batch(batch_size)
train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).batch(batch_size)

# valid_loader = valid_dataset.map(preprocess).repeat(epochs).batch(batch_size)
valid_loader = valid_dataset.map(preprocess).batch(batch_size)


# for img, lbl in train_loader.take(1):
#     print(img.shape)
#     print(lbl.shape)

train_step = len(train_loader)
val_step = len(valid_loader)

# train_step = (60000 // 32) + 1
# val_step = (10000 // 32) + 1

loss_dict = {}
val_loss_dict = {}

for epoch in range(1, epochs + 1):

    loss_list = []   
    for train_step_idx, (img, label) in enumerate(train_loader):
        model_params = model.trainable_variables
        
        with tf.GradientTape() as tape:
            out = model(img)
            loss = loss_fn(out, label)
            loss_list.append(loss.numpy().sum())

        grads = tape.gradient(loss, model_params)
        optimizer.apply_gradients(zip(grads, model_params))

        if ((train_step_idx+1) % 100 == 0):
            print(f"Epoch [{epoch}/{epochs}] Step [{train_step_idx + 1}/{train_step}] Loss: {loss.numpy().sum():.4f}")
    
    loss_dict[epoch] = loss_list
     
    val_loss_list = []
    for val_step_idx, (val_img, val_label) in enumerate(valid_loader):

        val_out = model(val_img)        
        val_loss = loss_fn(val_out, val_label)        
        val_loss_list.append(val_loss.numpy().sum())
        
    val_loss_dict[epoch] = val_loss_list
    
    print(f"Epoch [{epoch}] Train Loss: {mean(loss_dict[epoch]):.4f} Val Loss: {mean(val_loss_dict[epoch]):.4f}")
    print("========================================================================================")


    model.save_weights(f'checkpoint/cifar10_model_epoch_{epoch}.ckpt', save_format='tf')

model.save_weights('model/cifar10_model', save_format='tf')

print('model saved')
