import tensorflow as tf
from tensorflow.python.client import device_lib
import statistics

print(device_lib.list_local_devices())
# from tensorflow.keras.datasets import fashion_mnist
# from ten

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_test = (x_train / 255.0).astype('float32'), (x_test / 255.0).astype('float32')


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.Dense(256, activation='relu', input_shape=(784,))
        self.hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out
    
model = Model()

# tf.keras.utils.plot_model(model, show_shapes=True)
# model.summary()

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(lr=1e-4)

batch_size = 32

loss_dict = {}   

for epoch in range(5):
    loss_list = []   

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
        
    loss_dict[epoch] = statistics.mean(loss_list)
    
    print(loss_dict[epoch])