import tensorflow as tf
from tensorflow.keras.layers import *

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = Conv2D(filters=16, kernel_size=3, padding='same',
                                           strides=1, use_bias=False)
        self.batchnorm = BatchNormalization(trainable=True)
        self.relu = Activation('relu')
        self.avg_pool = AveragePooling2D(pool_size=(8, 8))
        self.flatten = Flatten()
        self.fc = Dense(10)
        self.softmax = Activation('softmax')

    def call(self, x):
        out0 = self.conv(x)
        print(out0) # debugging
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = self.flatten(out3)
        out5 = self.fc(out4)
        outputs = self.softmax(out5)
       
        return outputs

model = SimpleModel()

batch_size = 32
epochs = 3

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=0.001)

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

train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(3).batch(32)

valid_loader = valid_dataset.map(preprocess).repeat(3).batch(32)

for epoch in range(1, epochs + 1): 
    for img, label in train_loader:
        model_params = model.trainable_variables

        with tf.GradientTape() as tape:
            out = model(img)
            loss = loss_fn(out, label)

        grads = tape.gradient(loss, model_params)
        optimizer.apply_gradients(zip(grads, model_params))

    print(f"[{epoch}/{epochs}] finished")
    print('==================')

model.save_weights('cifar10_model', save_format='tf')

print('model saved')












# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# # tf.debugging.set_log_device_placement(True)