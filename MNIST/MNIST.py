import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import statistics
#import tqdm

print(device_lib.list_local_devices())
# from tensorflow.keras.datasets import fashion_mnist
# from ten

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = x_train / np.float32(255.0), x_test / np.float32(255.0)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.Dense(256, activation='relu', input_shape=(784,))
        self.hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        inputs = self.input_layer(x)
        hidden = self.hidden_layer(inputs)
        outputs = self.output_layer(hidden)
        return outputs
    
model = Model()

# InTensor = x_train[0, :, :]
# InTensor = InTensor.reshape(-1, 28*28)
# oot= model(InTensor)

# tf.keras.utils.plot_model(model, show_shapes=True)
# model.summary()

# model_t = Model(x_test, y_test)

# model_t.summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(lr=1e-4)

batch_size = 32

loss_dict = {}
t_loss_dict = {}

for epoch in range(5):
    loss_list = []   
    t_loss_list = []
    
    for i in range(x_train.shape[0] // batch_size):
        
        x_batch = x_train[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train[i * batch_size:(i + 1) * batch_size]
        
        x_batch = x_batch.reshape(-1, 28*28)
        
        y_batch = tf.one_hot(y_batch, 10)
        
        model_params = model.trainable_variables
        
        with tf.GradientTape() as tape:
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
        
            loss_list.append(loss.numpy().sum())
            
        grads = tape.gradient(loss, model_params)
    
        optimizer.apply_gradients(zip(grads, model_params))
        
        x_batch_test = x_test[i * batch_size:(i + 1) * batch_size]
        y_batch_test = y_test[i * batch_size:(i + 1) * batch_size]
        
        x_batch_test = x_batch_test.reshape(-1, 28*28)
        
        y_batch_test = tf.one_hot(y_batch_test, 10)
        
        t_out = model(x_batch_test)
        
        t_loss = loss_fn(t_out, y_batch_test)
        
        t_loss_list.append(t_loss.numpy().sum())
        
    loss_dict[epoch] = statistics.mean(loss_list)
    t_loss_dict[epoch] = statistics.mean(t_loss_list)
    
    print(loss_dict[epoch])
    print(t_loss_dict[epoch])
    print('==================')
    
    
    

    
    
