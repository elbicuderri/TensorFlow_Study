import tensorflow as tf
from tensorflow.keras.layers import *

class Rnn(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def call(self, input, hidden):
        combined = concatenate([input, hidden])
        next_hidden = Dense(self.hidden_size, activation='tanh')(combined)
        output = Dense(self.output_size, activation='softmax')(combined)
        return output, next_hidden
    
    def inithidden(self):
        return tf.zeros([1, self.hidden_size])