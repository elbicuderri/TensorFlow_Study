import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.python.client import device_lib
from statistics import mean

print(device_lib.list_local_devices())

class ConvTransposeModel(tf.keras.Model):
    def __init__(self):
        super(ConvTransposeModel, self).__init__()
        
        self.trainable = tf.constant(True)
        
        self.relu = Activation('relu')
        
        self.batchnorm1 = BatchNormalization(trainable=self.trainable, epsilon=0.001)
        
        self.batchnorm2 = BatchNormalization(trainable=self.trainable, epsilon=0.001)
        
        self.batchnorm3 = BatchNormalization(trainable=self.trainable, epsilon=0.001)
        
        self.batchnorm4 = BatchNormalization(trainable=self.trainable, epsilon=0.001)
        
        self.conv1 = Conv2D(filters=3,
                            kernel_size=3,
                            padding='same',
                            strides=2,
                            use_bias=False)
        
        self.conv2 = Conv2D(filters=4,
                            kernel_size=3,
                            padding='same',
                            strides=2,
                            use_bias=False)
        
        
        self.convtranspose1 = Conv2DTranspose(filters=3,
                                              kernel_size=3,
                                              padding='same',
                                              strides=2,
                                              use_bias=False)
        
        self.convtranspose2 = Conv2DTranspose(filters=10,
                                        kernel_size=3,
                                        padding='same',
                                        strides=2,
                                        use_bias=False)
        
        self.global_avg_pool = GlobalAveragePooling2D()
        
        self.flatten = Flatten()
        
        self.fc = Dense(10)
        
        self.softmax = Activation('softmax')
        
    def call(self, x):
        conv1 = self.conv1(x)
        conv1_batchnorm = self.batchnorm1(conv1)
        conv1_batchnorm_relu = self.relu(conv1_batchnorm) # (14, 14, 3)
        
        conv2 = self.conv2(conv1_batchnorm_relu)
        conv2_batchnorm = self.batchnorm2(conv2)
        conv2_batchnorm_relu = self.relu(conv2_batchnorm) # (7, 7, 4)
        
        convtranspose1 = self.convtranspose1(conv2_batchnorm_relu)
        convtranspose1_batchnorm = self.batchnorm3(convtranspose1)
        convtranspose1_batchnorm_relu = self.relu(convtranspose1_batchnorm) # (14, 14, 3)

        convtranspose2 = self.convtranspose2(convtranspose1_batchnorm_relu)
        convtranspose2_batchnorm = self.batchnorm4(convtranspose2)
        convtranspose2_batchnorm_relu = self.relu(convtranspose2_batchnorm) # (28, 28, 10)
        
        global_avg_pool = self.global_avg_pool(convtranspose2_batchnorm_relu) # (10, )
        
        flatten = self.flatten(global_avg_pool)
        
        dense = self.fc(flatten)
        
        logit = self.softmax(dense)
        
        return logit
    
    def model(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
model = ConvTransposeModel()

model.model().summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

print("Model Ready")

# ==================================================================================================== #

(x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

# ==================================================================================================== #
def preprocess(x, y):
    image = tf.reshape(x, [28, 28, 1])
    image = tf.cast(image, tf.float32) / 255.0
    # image = (image - 0.5) / 0.5

    label = tf.one_hot(y, depth=10)
    label = tf.squeeze(label)

    return image, label
# ==================================================================================================== #


batch_size = 32
epochs = 5

train_loader = train_dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).batch(batch_size)
valid_loader = valid_dataset.map(preprocess).batch(batch_size)

train_step = len(train_loader)
val_step = len(valid_loader)

loss_dict = {}
val_loss_dict = {}

acc_dict = {}
val_acc_dict = {}

for epoch in range(1, epochs + 1):

    loss_list = []
    num_correct = 0
    num_samples = 0
    
    for train_step_idx, (img, label) in enumerate(train_loader):
        model_params = model.trainable_variables
        model.trainable = tf.constant(True)
        
        with tf.GradientTape() as tape:
            out = model(img)
            
            predictions = np.argmax(out.numpy(), axis=1)
            num_correct += (predictions == np.argmax(label, axis=1)).sum()
            num_samples += predictions.shape[0]
            
            loss = loss_fn(out, label)
            loss_list.append(loss.numpy().sum())

        grads = tape.gradient(loss, model_params)
        optimizer.apply_gradients(zip(grads, model_params))

        if ((train_step_idx+1) % 100 == 0):
            print(f"Epoch [{epoch}/{epochs}] Step [{train_step_idx + 1}/{train_step}] Loss: {loss.numpy().sum():.4f}")
    
    loss_dict[epoch] = loss_list
    acc_dict[epoch] = (num_correct / num_samples) * 100
    
    val_loss_list = []
    val_num_correct = 0
    val_num_samples = 0
    
    for val_step_idx, (val_img, val_label) in enumerate(valid_loader):
        model.trainable = tf.constant(False)
        
        val_out = model(val_img)
        
        val_predictions = np.argmax(val_out.numpy(), axis=1)
        val_num_correct += (val_predictions == np.argmax(val_label, axis=1)).sum()
        val_num_samples += val_predictions.shape[0]
        
        val_loss = loss_fn(val_out, val_label)        
        val_loss_list.append(val_loss.numpy().sum())
        
    val_loss_dict[epoch] = val_loss_list
    val_acc_dict[epoch] = (val_num_correct / val_num_samples) * 100
    
    print(f"Epoch [{epoch}] Train Loss: {mean(loss_dict[epoch]):.4f} Val Loss: {mean(val_loss_dict[epoch]):.4f}")
    print(f"Epoch [{epoch}] Train Accuracy: {acc_dict[epoch]:.4f} Val Accuracy: {val_acc_dict[epoch]:.4f}")
    print("========================================================================================")

model.save_weights('model/cifar10_model', save_format='tf')

print('model saved')