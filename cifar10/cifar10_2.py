import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib
import statistics
#import tqdm

print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_loader = train_dataset.batch(32)

# for something in train_loader.take(1):
#     print(something)

def conv33(filters, kernel_size, strides, padding='same', use_bias=False):
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=use_bias
    )

class SimpleResNet(tf.keras.Model):
    def __init__(self):
        super(SimpleResNet, self).__init__()
    
        # self.conv0 = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation('relu')
        # ])

        self.conv =  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1, use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization(trainable=True)
        self.relu = tf.keras.layers.Activation('relu')
    
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # out0 = self.conv0(x)
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = self.flatten(out3)
        out5 = self.fc(out4)

        return out5

model = SimpleResNet()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

batch_size = 32

# loss_dict = {}

# with tf.device('/GPU:0'):
for epoch in range(1, 6):
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
    print(f"[{epoch}/5] finished")
    print('==================')

    if (epoch == 5):
        model.save_weights('model/cifar10_model_5', save_format='tf')

# weights = model.weights

# for i, w in enumerate(weights):
#     np.save(f"model/cifar10_{i}.npy", w.numpy())

model.save_weights('model/cifar10_model', save_format='tf')

# a = model.weights

# a = np.array(a)

# np.save('model/weight_arr.npy', a)

# try:
#     a.tofile('model/weights.wts', sep=",")
# except:
#     pass

# np.tofile(a, 'model/weights.wts')

print('model saved')
