import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense

print(K.image_data_format())

# Placeholder
input_tensor = K.placeholder(shape=(None, None, 3), ndim=3, dtype='float32')

# Variable: name에 공백이 있으면 안된다.
# 안의 값을 보고 싶으면 K.eval
var0 = K.eye(size=3)
var1 = K.zeros(shape=(3, 3))
var2 = K.ones(shape=(3, 3))
var3 = K.variable(value=np.random.random((224, 224, 3)),
                  dtype='float32', name='example_var', constraint=None)
var4 = K.constant(value=np.zeros((2, 2)), dtype='int32', shape=(2, 2), name='ex_constant')

var5 = K.variable(value=np.array([[2,2], [4,3]]), dtype='float32')
K.eval(var5)


# 따라하기
arr = np.array([[1,2], [3,4]])
var6 = K.ones_like(arr, dtype='float32')
var7 = K.zeros_like(arr, dtype='int32')
# var8 = K.identity(arr, name='identity')


# 텐서 조작: 랜덤 초기화 = Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(64, 64, 3), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(64, 64, 3), mean=0, scale=1) # Gaussian distribution

# Tensor Arithmetic
a = b * K.abs(c)
# c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
a = K.concatenate([b, c], axis=0)