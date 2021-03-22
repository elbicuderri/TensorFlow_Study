import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Dense, Activation

class LSTMCell(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.ii_layer = Dense(self.hidden_size)
        self.hi_layer = Dense(self.hidden_size)
        self.if_layer = Dense(self.hidden_size)
        self.hf_layer = Dense(self.hidden_size)
        self.ig_layer = Dense(self.hidden_size)
        self.hg_layer = Dense(self.hidden_size)
        self.io_layer = Dense(self.hidden_size)
        self.ho_layer = Dense(self.hidden_size)
        
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        
        self.dot = tf.math.multiply
        
    def call(self, input, hidden, cell):
        
        i = self.sigmoid(self.ii_layer(input) + self.hi_layer(hidden))
        f = self.sigmoid(self.if_layer(input) + self.hf_layer(hidden))
        g = self.tanh(self.ig_layer(input) + self.hg_layer(hidden))
        out = self.sigmoid(self.io_layer(input) + self.ho_layer(hidden))     
        
        # cell_state = tf.math.multiply(f, cell) + tf.math.multiply(i, g)
        # hidden_state = tf.math.multiply(out, self.tanh(cell_state))
        
        cell_state = self.dot(f, cell) + self.dot(i, g)
        hidden_state = self.dot(out, self.tanh(cell_state))
        
        return out, hidden_state, cell_state

## not implemented yet
class LSTM(tf.keras.Model): 
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.bidirectional = bidirectional
        
        self.lstm_cell = LSTMCell(self.input_size, self.hidden_size)
        
    def call(self, input):
        seq_len = input.shape[0]  
        batch = input.shape[1]
        
        # outs = tf.zeros([seq_len, batch, self.hidden_size]) # EagerTensor cannot assign and slice
        outs = np.zeros((seq_len, batch, self.hidden_size))
        
        for l in range(seq_len):
            out_sum = tf.zeros([1, batch, self.hidden_size])
    
            for n in range(self.num_layers):
                if (n == 0):
                    h_pre = tf.zeros([1, batch, self.hidden_size])
                    c_pre = tf.zeros([1, batch, self.hidden_size])

                    o_n, h_n, c_n = self.lstm_cell(input[l,:,:], h_pre, c_pre)
                    assert (o_n.shape == (1, batch, self.hidden_size) and 
                            h_n.shape == (1, batch, self.hidden_size) and
                            c_n.shape == (1, batch, self.hidden_size))
                    
                else:
                    o_n, h_n, c_n = self.lstm_cell(input[l,:,:], h_pre, c_pre) # output (1, batch, hidden_size)
                    assert (o_n.shape == (1, batch, self.hidden_size) and 
                            h_n.shape == (1, batch, self.hidden_size) and
                            c_n.shape == (1, batch, self.hidden_size))
                    
                    h_pre, c_pre = h_n, c_n
            
                out_sum += o_n 
                
            outs[l, :, :] = out_sum
                                                     
        return outs
    
seq_len = 100
batch = 16
input_size = 16
hidden_size = 32
num_layers = 16

InTensor = tf.random.uniform([seq_len, batch, input_size])

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

out = model(InTensor)

model.summary()

print(out.shape)